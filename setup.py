################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

# python setup.py build_ext --inplace

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import shutil
import os

build_folder = os.path.join(os.getcwd(), 'kernels')
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='shadowkv',
    ext_modules=[
        CUDAExtension(
            name='shadowkv',
            sources=[
                'kernels/main.cu',
                'kernels/rope.cu',
                'kernels/rope_new.cu',
                'kernels/gather_copy.cu',
                'kernels/batch_gather_gemm.cu',
                'kernels/batch_gemm_softmax.cu',
            ],
            include_dirs=[
                f'{current_dir}/3rdparty/cutlass/include',
                f'{current_dir}/3rdparty/cutlass/examples/common',
                f'{current_dir}/3rdparty/cutlass/tools/util/include',
                f'{current_dir}/kernels'
            ],
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': ['-std=c++17', '--expt-relaxed-constexpr', '-lcuda', '-lcudart'],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)

for so_file in (f for f in os.listdir(current_dir) if f.endswith('.so')):
    shutil.move(so_file, os.path.join(build_folder, so_file))