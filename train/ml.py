import os

for i in range(100):
    os.system("CUDA_VISIBLE_DEVICES=3 python train_cla.py")