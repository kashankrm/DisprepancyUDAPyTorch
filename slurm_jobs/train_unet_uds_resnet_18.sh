#!/bin/bash
### Name of the Jobs for the queueing-system
#SBATCH --job-name=train_deeplab
### Number of Nodes, for Abaqus one node with a maximum of 28 cores is enough
##SBATCH --nodes=1
### Number of tasks per node. This has to be the same a given in the Abaqus command after -cpus.
### A large number of cores will consume a lot of tokens. # We need to check how the number of cores needs to be specified. What the limitations are on the gpu
#SBATCH --ntasks-per-node=8
### Duration, in this case not necessary
###SBATCH --time=d-hh:mm:ss
### The queue (partition) should be abaqus --> changed it to gpu for our purpose
### #SBATCH --partition=abafast
### #SBATCH --partition=short
#SBATCH --partition=4gpu

### Get mail when certain events occur (BEGIN, END, ALL)
#SBATCH --mail-type END
#SBATCH --mail-user=muhammad.karim@iwm.fraunhofer.de
### Redirection of Standard Output and Error Messages, %j will be the SLURM-jobID of the job
#SBATCH -o slurm_logs/gpu_job."%j".out
#SBATCH -e slurm_logs/gpu_job."%j".err

# specify filename of python training script
cd ..
jobFile=train_unet_target
JOBNAME=DL_${jobFile}_${SLURM_JOBID}


# this defines the temporary scratch folder path
USERNAME=kari
JOBTMP=/scratch/${USERNAME}_${SLURM_JOB_ID}
export TMPDIR=${JOBTMP}  # scratch-directory for the job 

## initialize the conda environment inside /isi/w/lb27
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !! ;)
__conda_setup="$('/isi/w/lb27/softwares/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/isi/w/lb27/softwares/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/isi/w/lb27/softwares/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/isi/w/lb27/softwares/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export WKHTMLTOPDF_PATH=/isi/w/lb27/softwares/wkhtmltopdf/usr/local/bin/wkhtmltopdf
## activate the requried conda environment and run the python script
conda activate detectron
#python ${jobFile}.py --exp-name test_S_to_T3_K0_After_Aurele
python ${jobFile}.py --exp-name unet_vanilla_uds \
--gpu-id 0 \
--batch-size 12 \
--model-arch Unet \
--num-iterations 100000 \
--val-every-it 200 \
--data-dir-image /isi/w/lb27/data/PAG_segmentation/processed/semantic_segmentation/real_data/nital_pag_dataset_noset/images \
--data-dir-label /isi/w/lb27/data/PAG_segmentation/processed/semantic_segmentation/real_data/nital_pag_dataset_noset/masks \
--data-list-train /isi/w/lb27/data/PAG_segmentation/processed/semantic_segmentation/real_data/nital_pag_dataset_noset/perc_val_const/85/train_list.txt \

sleep 5
### This loop is required for the function above to work. As long as the lockfile exists this script will trap the TERM signal sent by the queueing-system.
    while [ -f ${JOBNAME}.lck ]; do
       sleep 5
    done
