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

conda env remove -n ShadowKV -y # remove the existing environment
conda create -n ShadowKV python=3.10 -y
conda activate ShadowKV
which python


# nemo dependencies
pip install wheel
pip install Cython
pip install youtokentome
pip install nemo_toolkit[all]==1.23 # make sure torchaudio is compiled with same CUDA version as torch

pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# flashinfer
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
pip install huggingface_hub==0.23.2

# build ShadowKV kernels
python setup.py build_ext --inplace