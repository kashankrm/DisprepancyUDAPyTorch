import argparse
from datetime import datetime

import optuna
from optuna.trial import TrialState
from trainer import AdversarialDATrainer, DiscrepencyDATrainer
from datasets import CustomDataset
import torch, os 
from torch.utils import data
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from augmentations import get_augmentations, get_basic_train_augment,get_basic_val_augment
from models import FCDiscriminator, UnetPlusPlus, add_feature_extractor, get_encoder_out_shape
from losses import MMD_loss, get_loss_func
from metrics import (
    dummy_metric,
    metric_wrapper,
    iou_metric_bg,
    iou_metric_fg,
    )
from defaults import *
from utils import get_preprocessing

from torch.utils.tensorboard import SummaryWriter

def optuna_objective(args,trial):
    
    adv_wth = trial.suggest_float("adv_wth", 0.0001, 0.01, log=False)
    # target_wth = trial.suggest_float("target_wth", 0.2, 0.8, log=False)
    discrepancy_level = trial.suggest_categorical("discrepancy_level", ["1", "2", "3", "4", "5"])
    # lr_d = trial.suggest_float("lr_d", 1e-6, 1e-2, log=True)
    hpo_data = {
        "adv_wth":adv_wth,
        # "target_wth":target_wth,
        "discrepancy_level":discrepancy_level,
    }
    print(f"trying {hpo_data}")
    args.discrepancy_weight = adv_wth
    # args.target_weight = target_wth
    args.discrepancy_level = discrepancy_level
    args.num_iterations = 10000
    return main(args,trial)

def main(args,trial=None):
    scratch_dir = os.environ['TMPDIR'] if 'TMPDIR' in os.environ else None
    train_aug, val_aug = get_augmentations(args.aug_type)
    train_loader = data.DataLoader(
            CustomDataset(
            args.data_dir_image, 
            args.data_list_train, 
            augment=train_aug,
            scratch_dir=scratch_dir,
            # preprocessing=get_preprocessing(preprocess_input),
            ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    if args.data_list_validation:
        val_loader = data.DataLoader(
                CustomDataset(
                args.data_dir_image, 
                args.data_list_validation, 
                augment=val_aug,
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
            augment=val_aug,
            scratch_dir=scratch_dir,
            # preprocessing=get_preprocessing(preprocess_input),
            ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    target_train_loader = data.DataLoader(
            CustomDataset(
            target_image_path, 
            target_list_train, 
            augment=train_aug,
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
            augment=train_aug,
            scratch_dir=scratch_dir,
            # preprocessing=get_preprocessing(preprocess_input),
            ),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    num_classes_info = get_encoder_out_shape(args)
    
    logdir = f'log_output/{args.exp_name}'


    cur_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    tblogger = SummaryWriter(log_dir=f"{logdir}/{cur_time}")
    # logger = CustomLogger(logdir)
    device = torch.device('cpu') if args.cpu else torch.device(f"cuda:{args.gpu_id}")
    model_cl = getattr(smp,args.model_arch)
    model = model_cl(args.encoder,encoder_weights=args.encoder_weights,classes=args.num_classes)
    add_feature_extractor(model)
    adv_model = FCDiscriminator(num_classes=num_classes_info[args.discrepency_level])
    
    opt = torch.optim.Adam(model.parameters(),lr=0.0006)
    adv_opt = torch.optim.Adam(adv_model.parameters(), lr=4e-4, betas=(0.9, 0.99))
    critera = get_loss_func(args.loss_func)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    trainer = AdversarialDATrainer(
        model,
        opt,
        args.num_iterations,
        device,
        tblogger,
        val_every_it=args.val_every_it,
        print_train_every_iter=args.print_train_every_it,
        trial=trial
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
    
    trainer.assign_criteria_to_dataset('train','source_train','seg_loss',1 - args.target_weight)
    trainer.assign_adv_criteria_to_dataset('source_train','target_train','adv','bce_loss',args.discrepency_level,args.discrepancy_weight)
    
    
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
        trainer.assign_criteria_to_dataset('train','semi_sup_dt','seg_loss',args.target_weight)
        trainer.assign_metric_to_dataset('semi_sup_dt','fg_iou')
        trainer.assign_metric_to_dataset('semi_sup_dt','bg_iou')
        
    train_dict = trainer.train()


    
    print("")
    return (train_dict['val_metrics'][-1]["target_val__fg_iou"]+train_dict['val_metrics'][-1]["target_val__bg_iou"])/2


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
    parser.add_argument("--target-weight", type=float,default= 0.4443572306536943,
                        help="weight for target seg loss while source weight is 1 - this value.")
    parser.add_argument("--discrepancy-weight", type=float,default= 0.001,
                        help="weight for discrepancy weight")
    parser.add_argument("--hpo",action="store_true", default=False, 
                        help="hyperparameter optimization over batch size, learning rate, momentum and weight decay")
    args = parser.parse_args()
    HPO = args.hpo

    if not HPO:
        
        main(args)

    else:

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda x: optuna_objective(args,x), n_trials=25)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))