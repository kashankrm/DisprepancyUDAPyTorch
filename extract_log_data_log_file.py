import os
from os import path as osp
from collections import defaultdict
import json,csv
import sys
def get_root():
    abspath = osp.abspath(".")
    abspath_b = abspath.split("/")
    abspath_b = abspath_b[:abspath_b.index("UDA_training")+1]
    return "/".join(abspath_b)
def remove_cuda_tensor(ln):
    ln = ln.replace('tensor(',"")
    for i in range(4):
        ln = ln.replace(f", device='cuda:{i}')","")
    return ln
def save_csv(x,y,k,log_path):
    try:
        with open(f"{log_path}/{k}.csv","w+") as f:
            
            writer = csv.writer(f)
            writer.writerow(["step",k])
            for x_v,y_v in zip(x,y):
                writer.writerow([x_v, y_v])
    except FileNotFoundError as e:
        print(f"Warning: {job_id} -> {log_path} folder does not exist")
    return
def LD_to_DL(ld):
    dl = defaultdict(list)
    for d in ld:
        for key,val in d.items():
            dl[key].append(val)
    return dl
def main(job_id):
    log_file = f"{get_root()}/slurm_jobs/slurm_logs/gpu_job.{job_id}.out" #sys.argv[1]
    print(f"processing jobno {job_id}")
    with open(log_file,"r+") as f:
        log = f.read()
    log = log.split("\n")
    '''
    starting training for log_output/unet_vanilla_uds_dense201_noleak_Dice/03_20_2023_18_21_16
    '''
    try:
        log_path = next(l for l in log if l.startswith("starting training for")).split(" ")[3]
    except :
        print(f"could not find log_path for file {log_file}, exiting now ...")
        return
    
    log_path = f"{get_root()}/{log_path}"
    if any([ld.endswith("csv") for ld in os.listdir(log_path)]):
        print(f"log_path {log_path} already contains csv's exiting now ...")
        return
    
    log = log[3:]
    train_scalers = []
    val_scalers = []
    for l in log:
        if "val_losses" in l:
            metric = json.loads(remove_cuda_tensor(l[l.index(', val_metrics ')+len(', metrics '):]).replace("'",'"'))
            l_split = l.split(" ")
            iteration = l_split[0].split("/")[0]
            data = {"step":int(iteration)}
            for k in metric:
                data[f"val_metric_{k}"] = metric[k]
            val_scalers.append(data)
        else:
            metric = json.loads(remove_cuda_tensor(l[l.index(', metrics ')+len(', metrics '):]).replace("'",'"'))
            losses = json.loads(remove_cuda_tensor(l[l.index(' losses ')+len(" losses "):l.index(', metrics ')]).replace("'",'"'))
            
            l_split = l.split(" ")
            iteration = l_split[0].split("/")[0]
            data = {"step":int(iteration)}
            for k in metric:
                data[f"metric_{k}"] = metric[k]
            for k in losses:
                data[f"loss_{k}"] = losses[k]
            train_scalers.append(data)
    train_scalers = LD_to_DL(train_scalers)
    val_scalers = LD_to_DL(val_scalers)
    for k in train_scalers:
        if k != 'step':
            save_csv(train_scalers['step'],train_scalers[k],k,log_path)
    for k in val_scalers:
        if k != 'step':
            save_csv(val_scalers['step'],val_scalers[k],k,log_path)
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        job_id = sys.argv[1]

        main(job_id)
    else:
        job_ids = os.listdir(f"{get_root()}/slurm_jobs/slurm_logs/")
        job_ids = [ji.split(".")[1] for ji in job_ids if ji.endswith(".out")]
        
        for ji in job_ids: main(ji)

