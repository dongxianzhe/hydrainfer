# dxz
a llm inference engine for academic research

## Getting Started

You can install it from source code.
```
conda create -n dxz_dev python=3.10
conda activate dxz_dev
pip install -r requirements.txt
git submodule init
git submodule update
pip install -e .
```

If you want it to run faster, you may try to install the following library
```
pip install flashinfer==0.1.6+cu124torch2.4 -i https://flashinfer.ai/whl/cu124/torch2.4
pip install flash-attn==2.7.0.post2
```

You can uninstall it with the following code:
```
pip uninstall dxz
conda deactivate
conda env remove -n dxz_dev
```

You can run offline benchmark with the following code:
```
cd benchmark
python offline_inference_vision_language_benchmark.py --backend=dxz --num-prompts=10
```

You can run online benchmark with the following code:
```
cd benchmark
python dxz.entrypoint.api_server
python online_inference_vision_language_benchmark.py --backend=dxz --num-prompts=10
```