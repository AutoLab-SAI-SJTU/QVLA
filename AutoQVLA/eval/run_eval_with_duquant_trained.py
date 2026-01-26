"""
Run LIBERO evaluation with trained DuQuant parameters applied online (runtime layer replacement).

This is a small wrapper around OpenVLA's evaluation loop that:
- loads OpenVLA checkpoint
- injects DuQuant Quant*DecoderLayer using `duquant_parameters.pth`
- runs LIBERO evaluation

It also prints full tracebacks on exceptions to make failures debuggable.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tqdm

from libero.libero import benchmark

# Prefer the QVLM OpenVLA evaluation stack (this repo contains multiple "experiments" trees).
# This makes imports match the original scripts that use `from experiments...`.
_QVLM_OPENVLA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "QVLM", "openvla"))
if _QVLM_OPENVLA_ROOT not in sys.path:
    sys.path.append(_QVLM_OPENVLA_ROOT)

# OpenVLA eval utilities (QVLM tree)
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor, get_vla_action
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from AutoQVLAv2.inject_duquant_trained import DuQuantRuntimeConfig, apply_trained_duquant_to_language_model


@dataclass
class GenerateConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    num_tasks: Optional[int] = None
    max_steps_override: Optional[int] = None

    # Logging
    run_id_note: Optional[str] = "duquant-trained"
    local_log_dir: str = "./experiments/logs"
    seed: int = 7


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_checkpoint", type=str, required=True)
    p.add_argument("--task_suite_name", type=str, default="libero_spatial")
    p.add_argument("--num_trials_per_task", type=int, default=1)
    p.add_argument("--num_tasks", type=int, default=None, help="Optional: only evaluate first N tasks in the suite.")
    p.add_argument("--max_steps", type=int, default=None, help="Optional: override max_steps for a quick smoke run.")
    p.add_argument("--local_log_dir", type=str, default="./experiments/logs")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--duquant_parameters", type=str, required=True)
    p.add_argument("--gates_path", type=str, default=None)  # kept for CLI compatibility (mixed-bit handled elsewhere)

    # DuQuant runtime knobs (match the training command)
    p.add_argument("--wbits", type=int, default=4)
    p.add_argument("--abits", type=int, default=16)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--max_rotation_step", type=int, default=256)
    p.add_argument("--permutation_times", type=int, default=1)
    p.add_argument("--group_size", type=int, default=-1)
    p.add_argument("--symmetric", action="store_true", default=True)
    p.add_argument("--no_rotate", action="store_true", default=False)

    return p.parse_args()


def eval_libero_with_duquant(cfg: GenerateConfig, duquant_parameters: str, dq_cfg: DuQuantRuntimeConfig) -> None:
    assert cfg.pretrained_checkpoint is not None
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    model = get_model(cfg)
    processor = get_processor(cfg)

    # Inject trained DuQuant params
    model, applied = apply_trained_duquant_to_language_model(model, duquant_parameters, runtime_cfg=dq_cfg)
    print(f"[AutoQVLAv2][duquant-trained] applied {applied}/32 decoder layers from duquant_parameters")
    print("[AutoQVLAv2][duquant-trained] using preloaded model with trained DuQuant parameters")

    # Logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    resize_size = get_image_resize_size(cfg)

    total_episodes, total_successes = 0, 0
    num_tasks_to_run = num_tasks_in_suite if cfg.num_tasks is None else min(int(cfg.num_tasks), int(num_tasks_in_suite))
    for task_id in tqdm.tqdm(range(num_tasks_to_run)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []

            if cfg.max_steps_override is not None:
                max_steps = int(cfg.max_steps_override)
            else:
                if cfg.task_suite_name == "libero_spatial":
                    max_steps = 220
                elif cfg.task_suite_name == "libero_object":
                    max_steps = 280
                elif cfg.task_suite_name == "libero_goal":
                    max_steps = 300
                elif cfg.task_suite_name == "libero_10":
                    max_steps = 520
                elif cfg.task_suite_name == "libero_90":
                    max_steps = 400
                else:
                    max_steps = 300

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")

            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Force `use_cache=False` to avoid transformers Cache/DynamicCache code paths.
                    action = get_vla_action(
                        model,
                        processor,
                        str(cfg.pretrained_checkpoint),
                        observation,
                        task_description,
                        cfg.unnorm_key,
                        center_crop=cfg.center_crop,
                        use_cache=False,
                    )
                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    tb = traceback.format_exc()
                    print(tb)
                    log_file.write(tb + "\n")
                    log_file.flush()
                    break

            task_episodes += 1
            total_episodes += 1

            save_rollout_video(replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file)

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()


def main() -> None:
    args = _parse_args()
    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        num_tasks=args.num_tasks,
        max_steps_override=args.max_steps,
        local_log_dir=args.local_log_dir,
        seed=args.seed,
    )
    dq_cfg = DuQuantRuntimeConfig(
        wbits=args.wbits,
        abits=args.abits,
        block_size=args.block_size,
        max_rotation_step=args.max_rotation_step,
        permutation_times=args.permutation_times,
        group_size=args.group_size,
        symmetric=args.symmetric,
        no_rotate=args.no_rotate,
    )
    eval_libero_with_duquant(cfg, args.duquant_parameters, dq_cfg)


if __name__ == "__main__":
    main()


