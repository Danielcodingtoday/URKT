import subprocess
import time
import random

dataset=[ 'SFEW', 'FER2013', 'ExpW', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA',]
choice = dataset[2]   #选择数据集
mode='stage1'   #选择阶段1还是阶段2

# # 定义休眠时间列表
# sleep_times = [300, 180, 480, 600]  # 对应 5 分钟、3 分钟、8 分钟和 10 分钟
# # 随机选择一个休眠时间
# sleep_time = random.choice(sleep_times)
# # 进行休眠
# time.sleep(sleep_time)
# 执行命令获取显卡内存信息
while True:
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader'])    
    # 解析输出并提取显卡内存剩余值
    memory_free_list = [int(memory.strip().split()[0]) for memory in output.decode().strip().split('\n')]   
    # 寻找内存剩余最大值
    max_memory_free = max(memory_free_list)
    # 获取最大内存剩余对应的显卡索引
    gpu_id = memory_free_list.index(max_memory_free)

# python scripts/train.py -m DTransformer -d [assist09,assist17,algebra05,statics] -bs 32 -tbs 32 -p -cl --proj [-o output/DTransformer_assist09] [--device cuda]
    # 输出当前剩余内存最大的显卡和剩余内存大小
    print(f"当前剩余内存最大的显卡是 GPU:{gpu_id+1}，剩余内存：{max_memory_free} MiB")
    if max_memory_free > 100:
         if mode=='stage1':
               # 执行命令
               subprocess.run(['python', '/home/q22301155/codedemo/KT/Uncertain_V1/scripts/test.py',
                                '-m', 'DTransformer',
                                '-d','algebra05',
                                '-bs', '32',
                                 '-p',
                                '-f_c','/home/q22301155/codedemo/KT/Uncertain_V1/result/algebra05/best_model_c.pt',
                                '-f_uc','/home/q22301155/codedemo/KT/Uncertain_V1/result/algebra05/best_model_uc.pt',
                                '--device',f'cuda:{gpu_id}'
                                ]) 

         # 停止脚本
         break  
    
    sleep_times = [300, 180, 480, 600]  # 对应 5 分钟、3 分钟、8 分钟和 10 分钟
    # 随机选择一个休眠时间
    # sleep_time = random.choice(sleep_times)
    sleep_time = 10
    # 进行休眠
    time.sleep(sleep_time) 



