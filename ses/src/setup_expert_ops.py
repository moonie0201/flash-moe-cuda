"""Build script for expert_ops CUDA extension.

Usage:
    python setup_expert_ops.py build_ext --inplace

Or with pip:
    pip install -e . --no-build-isolation
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='expert_ops',
    ext_modules=[
        CUDAExtension(
            name='expert_ops',
            sources=['expert_ops.cu'],
            extra_compile_args={
                'nvcc': [
                    '-O3',
                    '-arch=sm_86',      # RTX 3060 (Ampere)
                    '--use_fast_math',
                    '-lineinfo',
                    '--ptxas-options=-v',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
