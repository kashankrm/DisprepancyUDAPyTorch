import copy
from datetime import datetime
from threading import Thread
import os.path as osp
import os
import json
import torch

class CustomLogger:
    def __init__(self,logdir) -> None:
        self.name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.log_path = osp.join(logdir,self.name)
        self.max_threads = 100
        self.threads = []
        os.makedirs(self.log_path)
    def sanitize(self,data):
        def convert_scaler(val):
            if isinstance(val,torch.Tensor):
                val = val.cpu().detach().numpy()
                return val.tolist() if val.size>1 else float(val)
            else:
                val
        data = copy.deepcopy(data)
        if isinstance(data,dict):
            for k,v in data.items():
                data[k] = convert_scaler(v)
            return data
        else:
            return convert_scaler(data)
    def add_scalers(self,it,phase,name,data):
        
        if name == "metrics":
            data = {k:self.sanitize(v) for k,v in data.items()}
        os.makedirs(osp.join(self.log_path,phase),exist_ok=True)
        file_name = f"{it}_{name}.json"
        with open(osp.join(self.log_path,phase,file_name),"w+") as f:
            json.dump(data,f)
    def add_scalers_dict(self,train_it,phase,ret_dict):
        for k,v in ret_dict.items():
            for i,v2 in enumerate(v):
                self.add_scalers(i,phase,k,v2)

    def metric_func(self,*args):
        metric_name,train_it,inputs,outputs,labels,im_name = args
        self.add_scalers(train_it,metric_name,)