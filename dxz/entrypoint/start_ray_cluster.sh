export CUDA_VISIBLE_DEVICES=2
ray stop
ray start --head

/home/xzd/anaconda3/envs/python3_10/bin/python ../../test.py