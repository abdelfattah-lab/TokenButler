## Environment Set Up
To reproduce the results in the paper, you need to set up the environment as follows with a single A100 GPU:
```bash
# create env
uv venv --python 3.11 && source .venv/bin/activate && uv pip install --upgrade pip

# install packages
uv pip install -r requirements.txt
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# flashinfer
uv pip install flashinfer-python==0.3.1

# cutlass
mkdir 3rdparty
git clone https://github.com/NVIDIA/cutlass.git 3rdparty/cutlass

# build kernels for ShadowKV
python setup.py build_ext --inplace

# install MInfernece
uv pip install minference
```
## Supported Models
Currently, we support the following LLMs:
- Llama-3.1-8B

## Accuracy Evaluations
Here we provide an example to build the dataset and run evaluation for the [RULER](https://github.com/hsiehjackson/RULER) benchmark with Llama-3-8B-1M.
