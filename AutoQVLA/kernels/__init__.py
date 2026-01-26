"""
Kernels for AutoQVLA-style low-bit GEMM.

Currently provides:
- W4A4 Triton kernel with 4-bit packed weights and on-the-fly 4-bit activation quantization.
"""


