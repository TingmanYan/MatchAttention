import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='match_attention',
    version='0.7',
    description='Match Attention CUDA Extension for PyTorch',
    author='TingmanYan',
    ext_modules=[
        CUDAExtension('match_attention', [
            'src/match_former_cuda.cpp',
            'src/match_former_cuda_kernel.cu',
            'src/match_former_fused_forward.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)