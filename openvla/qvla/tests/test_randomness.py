import pytest

from qvla import artifacts


def test_omitted_seed_uses_entropy_and_explicit_seed_is_preserved(monkeypatch):
    values = iter([101, 202])
    monkeypatch.setattr(artifacts.secrets, "randbits", lambda bits: next(values))
    assert artifacts.resolve_seed() == 101
    assert artifacts.resolve_seed() == 202
    assert artifacts.resolve_seed(0) == 0
    assert artifacts.resolve_seed(2**32 - 1) == 2**32 - 1


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_seed_outside_supported_range_is_rejected(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        artifacts.resolve_seed(seed)
