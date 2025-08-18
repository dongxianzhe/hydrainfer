# hydrainfer
a MLLM (Multimodal Large Language Models) inference engine for academic research.

## Getting Started

You can install it from source code.
```
conda create -n hydrainfer python=3.10
conda activate hydrainfer
pip install -r requirements.txt
git submodule init
git submodule update
pip install -e .

conda activate hydrainfer
conda install -c anaconda cmake=3.26
conda install -c conda-forge gcc_linux-64 gxx_linux-64 ninja
ORIGINAL_DIR=$(pwd)
cd $CONDA_PREFIX/bin
ln -sf x86_64-conda_cos7-linux-gnu-gcc gcc
ln -sf x86_64-conda_cos7-linux-gnu-g++ g++
cd $ORIGINAL_DIR
conda install -c nvidia/label/cuda-12.4.0 cuda
mkdir build
cd build
cmake .. -GNinja -DUSE_CXX11_ABI=OFF
cmake --build . --target block_migration activation position_embedding kv_cache_kernels cache_kernels norm moe
```

If you want it to run faster, you may try to install the following library, or you may skip them.
```
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.4/
pip install flash-attn==2.7.0.post2
```
If you want it to run faster, you may try to build the following library, or you may skip them.
```
cmake --build . --target flash_attn 
```


You can uninstall it with the following code:
```
pip uninstall hydrainfer
conda deactivate
conda env remove -n hydrainfer
```

Before testing inference correctness, it's necessary to set up the vLLM environment for dataset preprocessing.
```
conda create -n vllm python=3.12
conda activate vllm
pip install vllm==0.8.2
```

Dataset preprocess.
```
cd evaluation/dataset_analysis
./preprocess.sh
```

You can use script to test inference correctness. The inference results will be in the evaluation/test_correctness/result directory.:
```
cd evaluation/test_correctness
./evaluation.sh
```
# Citation
If you use hydrainfer for your research, please cite our [paper](https://arxiv.org/abs/2505.12658): 
```
@misc{dong2025hydrainferhybriddisaggregatedscheduling,
      title={HydraInfer: Hybrid Disaggregated Scheduling for Multimodal Large Language Model Serving}, 
      author={Xianzhe Dong and Tongxuan Liu and Yuting Zeng and Liangyu Liu and Yang Liu and Siyu Wu and Yu Wu and Hailong Yang and Ke Zhang and Jing Li},
      year={2025},
      eprint={2505.12658},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2505.12658}, 
}
```