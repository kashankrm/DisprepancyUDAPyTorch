from datetime import datetime
import os
from glob import glob
import shutil


def get_all_files(path,ext='.csv'):
    """
    Recursively search all directories for log files in the given path.
    
    Parameters:
    path (str): The path to list.
    
    Returns:
    list: A list of all log files.
    """
    log_files = []
    dir_list = []
    for root, dirs, files in os.walk(path):
        dir_list.extend(dirs)
        logs = [f for f in files if f.endswith(ext)]
        if len(logs)>0:
            logs = [os.path.join(root,l) for l in logs]
            log_files = log_files+ logs
    return log_files
def copy_file(inp_path,out_path):
    out_folder = os.path.join(*out_path.split('/')[:-1])
    os.makedirs(out_folder,exist_ok=True)
    if os.path.exists(out_path):
        if REPLACE_FILE:
            os.remove(out_path)
            shutil.copy2(inp_path,out_path)
    else:         
        shutil.copy2(inp_path,out_path)

def recent_job(model_name):

    all_versions = os.listdir(os.path.join(log_out,model_name))
    all_versions = [av for av in all_versions if os.path.isdir(os.path.join(log_out,model_name,av))]
    all_versions_dt = [(i,datetime.strptime(av,datetime_format)) for i,av in enumerate(all_versions)]
    all_versions_dt = list(reversed(sorted(all_versions_dt,key=lambda x:x[1])))
    chosen_av = all_versions_dt[RECENT_IND] if len(all_versions)>=(RECENT_IND+1) else all_versions_dt[-1]
    recent_job_path = all_versions[chosen_av[0]]
    return os.path.join(log_out,model_name,recent_job_path)
log_out = 'log_output'
copy_out = 'csv_output'
REPLACE_FILE =True
RECENT_IND = 0
model_names = os.listdir(log_out)
datetime_format = "%m_%d_%Y_%H_%M_%S"

if os.path.exists(copy_out):
    print(f"removing {copy_out}")
    shutil.rmtree(copy_out)
for mn in model_names:
    log_path = recent_job(mn)
    csv_files = get_all_files(log_path)
    for csv in csv_files:
        copy_file(csv,os.path.join(copy_out,mn,csv.split('/')[-1]))
    png_files = get_all_files(log_path,'.png')
    for png in png_files:
        copy_file(png,os.path.join(copy_out,mn,"images",png.split('/')[-1]))
