import argparse
from datetime import datetime
from trainer import DiscrepencyDATrainer
from datasets import CustomDataset
import torch, os 
from torch.utils import data
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from augmentations import get_basic_train_augment,get_basic_val_augment
from models import UnetPlusPlus
from losses import MMD_loss
from metrics import (
    dummy_metric,
    metric_wrapper,
    iou_metric_bg,
    iou_metric_fg,
    )
from defaults import *
from utils import get_preprocessing

from torch.utils.tensorboard import SummaryWriter
def main(args):
    scratch_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else None
    
    train_loader = data.DataLoader(
            CustomDataset(
            args.data_dir_image, 
            args.data_list_train, 
            augment=get_basic_train_augment(),
            scratch_dir=scratch_dir,
            # preprocessing=get_preprocessing(preprocess_input),
            ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    if args.data_list_validation:
        val_loader = data.DataLoader(
                CustomDataset(
                args.data_dir_image, 
                args.data_list_validation, 
                augment=get_basic_val_augment(),
                scratch_dir=scratch_dir,
                # preprocessing=get_preprocessing(preprocess_input),
                ),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    target_image_path = args.target_data_dir_image#"/isi/w/lb27/data/PAG_segmentation/processed/semantic_segmentation/real_data/nital_pag_dataset_noset/images"
    target_list_train = args.target_data_list_train#"/isi/w/lb27/data/PAG_segmentation/processed/semantic_segmentation/real_data/nital_pag_dataset_noset/perc_val_const/5/val_list.txt"
    target_list_val = args.target_data_list_validation#"/isi/w/lb27/data/PAG_segmentation/processed/semantic_segmentation/real_data/nital_pag_dataset_noset/perc_val_const/5/val_list.txt"
    target_val_loader = data.DataLoader(
            CustomDataset(
            target_image_path, 
            target_list_val, 
            augment=get_basic_val_augment(),
            scratch_dir=scratch_dir,
            # preprocessing=get_preprocessing(preprocess_input),
            ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    target_train_loader = data.DataLoader(
            CustomDataset(
            target_image_path, 
            target_list_train, 
            augment=get_basic_val_augment(),
            scratch_dir=scratch_dir,
            # preprocessing=get_preprocessing(preprocess_input),
            ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    if args.semi_sup_target_list:
        semi_sup_data_list = args.semi_sup_target_list
        semi_sup_loader = data.DataLoader(
            CustomDataset(
            target_image_path, 
            semi_sup_data_list, 
            augment=get_basic_train_augment(),
            scratch_dir=scratch_dir,
            # preprocessing=get_preprocessing(preprocess_input),
            ),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    logdir = f'log_output/{args.exp_name}'

    cur_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    tblogger = SummaryWriter(log_dir=f"{logdir}/{cur_time}")
    # logger = CustomLogger(logdir)
    device = torch.device('cpu') if args.cpu else torch.device(f"cuda:{args.gpu_id}")
    model = UnetPlusPlus(args.encoder,encoder_weights=args.encoder_weights,classes=args.num_classes)
    opt = torch.optim.Adam(model.parameters(),lr=0.0006)
    critera = smp.losses.dice.DiceLoss(mode=smp.losses.MULTICLASS_MODE,ignore_index=255,smooth=0.5)
    mmd_loss = MMD_loss()
    trainer = DiscrepencyDATrainer(model,opt,args.num_iterations,device,tblogger,val_every_it=args.val_every_it,print_train_every_iter=args.print_train_every_it)
    trainer.add_dataset("source_train",train_loader)
    trainer.add_dataset("source_val",val_loader)
    trainer.add_dataset("target_train",target_train_loader)
    trainer.add_dataset("target_val",target_val_loader)
    
    trainer.add_criteria("seg_loss",critera)
    trainer.add_criteria("mmd_loss",mmd_loss)

    trainer.assign_disp_criteria_to_dataset('source_train','target_train','mmd_loss',args.discrepency_level,0.005)
    
    trainer.add_metric('fg_iou',iou_metric_fg)
    trainer.add_metric('bg_iou',iou_metric_bg)
    
    trainer.assign_criteria_to_dataset('train','source_train','seg_loss',0.5)
    
    
    trainer.assign_metric_to_dataset('source_train','fg_iou')
    trainer.assign_metric_to_dataset('source_train','bg_iou')
    
    trainer.assign_metric_to_dataset('target_val','fg_iou')
    trainer.assign_metric_to_dataset('target_val','bg_iou')
    
    trainer.assign_metric_to_dataset('source_val','fg_iou')
    trainer.assign_metric_to_dataset('source_val','bg_iou')

    # we want to tell trainer that multi_v5_val should be run at val phase
    trainer.assign_criteria_to_dataset('val','target_val',None)
    trainer.assign_criteria_to_dataset('val','source_val',None)
    if args.semi_sup_target_list:
        trainer.add_dataset("semi_sup_dt",semi_sup_loader)
        trainer.assign_criteria_to_dataset('train','semi_sup_dt','seg_loss')
        trainer.assign_metric_to_dataset('semi_sup_dt','fg_iou')
        trainer.assign_metric_to_dataset('semi_sup_dt','bg_iou')
        
    train_dict = trainer.train()
    

    print("")


if __name__ == '__main__':
    parser = get_arg_parser()
    
    parser.add_argument("--target-data-dir-image", type=str, required=True,
                        help="Path to the directory containing the target images.")
    parser.add_argument("--target-data-dir-label", type=str, required=False,
                        help="Path to the directory containing the target labels.")
    parser.add_argument("--target-data-list-train", type=str,required=True,
                        help="Path to the file listing the target images for training.")
    parser.add_argument("--semi-sup-target-list", type=str,default=SEMI_SUP_DATA_LIST,
                        help="Path to the file listing the target images for training.")
    parser.add_argument("--target-data-list-validation", type=str,required=False,
                        help="Path to the file listing the target images for training.")
    parser.add_argument("--discrepency-level", type=str,default="output",
                        help="on which feature level to perform discrepency (possible choices: output, 0-5).")
    args = parser.parse_args()
    main(args)