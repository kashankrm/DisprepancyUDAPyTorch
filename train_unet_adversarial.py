import argparse
from datetime import datetime
from trainer import AdversarialDATrainer, DiscrepencyDATrainer
from datasets import CustomDataset
import torch, os 
from torch.utils import data
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from augmentations import get_basic_train_augment,get_basic_val_augment
from models import FCDiscriminator, UnetPlusPlus
from losses import MMD_loss
from metrics import (
    dummy_metric,
    metric_wrapper,
    iou_metric_bg,
    iou_metric_fg,
    )
from defaults import get_arg_parser
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

    num_classes_info = {
        "output":args.num_classes,
        "0":3,
        "1":64,
        "2":64,
        "3":128,
        "4":256,
        "5":512,
        
    }
    
    logdir = f'log_output/{args.exp_name}'


    cur_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    tblogger = SummaryWriter(log_dir=f"{logdir}/{cur_time}")
    # logger = CustomLogger(logdir)
    device = torch.device('cpu') if args.cpu else torch.device(f"cuda:{args.gpu_id}")
    model = UnetPlusPlus(args.encoder,encoder_weights=args.encoder_weights,classes=args.num_classes)
    adv_model = FCDiscriminator(num_classes=num_classes_info[args.discrepency_level])
    
    opt = torch.optim.Adam(model.parameters(),lr=0.0006)
    adv_opt = torch.optim.Adam(adv_model.parameters(), lr=4e-4, betas=(0.9, 0.99))
    critera = smp.losses.dice.DiceLoss(mode=smp.losses.MULTICLASS_MODE,smooth=0.5)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    trainer = AdversarialDATrainer(
        model,
        opt,
        args.num_iterations,
        device,
        tblogger,
        val_every_it=args.val_every_it,
        print_train_every_iter=args.print_train_every_it
        )
    trainer.add_adv_model("adv",adv_model,adv_opt)

    trainer.add_dataset("source_train",train_loader)
    trainer.add_dataset("source_val",val_loader)
    trainer.add_dataset("target_train",target_train_loader)
    trainer.add_dataset("target_val",target_val_loader)
    
    trainer.add_criteria("seg_loss",critera)
    trainer.add_criteria("bce_loss",bce_loss)

    
    trainer.add_metric('fg_iou',iou_metric_fg)
    trainer.add_metric('bg_iou',iou_metric_bg)
    
    trainer.assign_criteria_to_dataset('train','source_train','seg_loss')
    trainer.assign_adv_criteria_to_dataset('source_train','target_train','adv','bce_loss',args.discrepency_level)
    
    
    trainer.assign_metric_to_dataset('source_train','fg_iou')
    trainer.assign_metric_to_dataset('source_train','bg_iou')
    
    trainer.assign_metric_to_dataset('target_val','fg_iou')
    trainer.assign_metric_to_dataset('target_val','bg_iou')
    
    trainer.assign_metric_to_dataset('source_val','fg_iou')
    trainer.assign_metric_to_dataset('source_val','bg_iou')

    # we want to tell trainer that multi_v5_val should be run at val phase
    trainer.assign_criteria_to_dataset('val','target_val',None)
    trainer.assign_criteria_to_dataset('val','source_val',None)
        
    train_dict = trainer.train()


    if not os.path.exists(f"{logdir}/{cur_time}"):
        os.makedirs(f"{logdir}/{cur_time}")
    torch.save(model.state_dict(), os.path.join(f"{logdir}/{cur_time}", 'end_model.pth'))
    print("")


if __name__ == '__main__':
    parser = get_arg_parser()
    
    parser.add_argument("--target-data-dir-image", type=str, required=True,
                        help="Path to the directory containing the target images.")
    parser.add_argument("--target-data-dir-label", type=str, required=False,
                        help="Path to the directory containing the target labels.")
    parser.add_argument("--target-data-list-train", type=str,required=True,
                        help="Path to the file listing the target images for training.")
    parser.add_argument("--target-data-list-validation", type=str,required=False,
                        help="Path to the file listing the target images for training.")
    parser.add_argument("--discrepency-level", type=str,default="output",
                        help="on which feature level to perform discrepency (possible choices: output, 0-5).")
    args = parser.parse_args()
    main(args)