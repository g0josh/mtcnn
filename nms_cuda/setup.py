from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nms_cuda',
    ext_modules=[
        CUDAExtension('nms_cuda', [
            'nms_cuda.cpp',
            'nms_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
})