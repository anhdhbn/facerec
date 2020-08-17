#!/usr/bin/env python
import psutil
import pickle
import nvidia_smi
from time import sleep
from sys import stdout


# gives a single float value
psutil.cpu_percent()
# gives an object with many fields
psutil.virtual_memory()
# you can convert that object to a dictionary 
dict(psutil.virtual_memory()._asdict())
# you can have the percentage of used RAM
# print(psutil.virtual_memory().percent)

nvidia_smi.nvmlInit()
handle_gpu1 = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
res_gpu1 = nvidia_smi.nvmlDeviceGetUtilizationRates(handle_gpu1)

handle_gpu0 = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
res_gpu0 = nvidia_smi.nvmlDeviceGetUtilizationRates(handle_gpu0)

while True:
    percent_ram = psutil.virtual_memory().percent
    percent_GPU_memory_0 = res_gpu0.memory
    percent_GPU_load_0 = res_gpu0.gpu
    
    percent_GPU_memory_1 = res_gpu1.memory
    percent_GPU_load_1 = res_gpu1.gpu

    print(f"""ram: {percent_ram}%, gpu[0]: {percent_GPU_load_0}%, gpu-mem[0]: {percent_GPU_memory_0}%, gpu[1]: {percent_GPU_load_1}%, gpu-mem[1]: {percent_GPU_memory_1}%""", end="\r")

    sleep(1)
    # print(f'ram: {percent_ram}%, gpu[1]: {percent_GPU_load}%, gpu-mem[1]: {percent_GPU_memory}%',sep=' ', end='', flush=True)


