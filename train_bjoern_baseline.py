from datetime import datetime
from trainer import Trainer
from datasets import CustomDataset
import torch, os 
from torch.utils import data
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from augmentations import get_basic_train_augment,get_basic_val_augment,bjoern_augmentation
from metrics import (
    dummy_metric,
    metric_wrapper,
    iou_metric_bg,
    iou_metric_fg,
    )
from defaults import get_arg_parser
from utils import get_preprocessing

from torch.utils.tensorboard import SummaryWriter


from losses import WJaccardLoss,WBCE,WDiceLoss
def bjoern_paper_loss():
    dice = WDiceLoss()
    jaccard = WJaccardLoss()
    wbce = WBCE()
    def func(input,target):
        return dice(input,target) + (0.5*jaccard(input,target)+0.5*wbce(input,target))
    return func

def main(args):
    scratch_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else None
    preprocess_input = get_preprocessing_fn(args.encoder, pretrained=args.encoder_weights)
    train_loader = data.DataLoader(
            CustomDataset(
            args.data_dir_image, 
            args.data_list_train, 
            augment=bjoern_augmentation(),
            scratch_dir=scratch_dir,
            # preprocessing=get_preprocessing(preprocess_input),
            ),
        batch_size=8, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    if args.data_list_validation:
        target_image_path = args.data_dir_image
        target_val_list = args.data_list_validation
    else:
        target_image_path = "/isi/w/lb27/data/PAG_segmentation/processed/semantic_segmentation/real_data/nital_pag_dataset_noset/images"
        target_val_list = "/isi/w/lb27/data/PAG_segmentation/processed/semantic_segmentation/real_data/nital_pag_dataset_noset/perc_val_const/5/val_list.txt"
    target_val_loader = data.DataLoader(
            CustomDataset(
            target_image_path, 
            target_val_list, 
            augment=get_basic_val_augment(),
            scratch_dir=scratch_dir,
            # preprocessing=get_preprocessing(preprocess_input),
            ),
        batch_size=8, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    logdir = f'log_output/{args.exp_name}'

    cur_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    tblogger = SummaryWriter(log_dir=f"{logdir}/{cur_time}")
    # logger = CustomLogger(logdir)
    device = torch.device('cpu') if args.cpu else torch.device(f"cuda:{args.gpu_id}")
    model = smp.Unet("densenet201",encoder_weights=args.encoder_weights,classes=args.num_classes)
    opt = torch.optim.Adam(model.parameters(),lr=0.001)
    critera = bjoern_paper_loss()
    trainer = Trainer(model,opt,args.num_iterations,device,tblogger,val_every_it=args.val_every_it)
    trainer.add_dataset("target_train",train_loader)
    
    trainer.add_dataset("target_val",target_val_loader)
    trainer.add_criteria("seg_loss",critera)
    trainer.add_metric('fg_iou',iou_metric_fg)
    trainer.add_metric('bg_iou',iou_metric_bg)
    trainer.assign_criteria_to_dataset('train','target_train','seg_loss')
    trainer.assign_metric_to_dataset('target_train','fg_iou')
    trainer.assign_metric_to_dataset('target_train','bg_iou')
    
    trainer.assign_metric_to_dataset('target_val','fg_iou')
    trainer.assign_metric_to_dataset('target_val','bg_iou')
    # we want to tell trainer that multi_v5_val should be run at val phase
    trainer.assign_criteria_to_dataset('val','target_val',None)
    
    train_dict = trainer.train()
    print("")


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()
    main(args)





