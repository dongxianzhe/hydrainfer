export CUDA_VISIBLE_DEVICES=2,3
ray stop
ray start --head

/home/xzd/anaconda3/envs/python3_10/bin/python api_server.py