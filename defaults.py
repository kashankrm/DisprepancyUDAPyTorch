import argparse

BATCH_SIZE = 12
NUM_WORKERS = 4
ENCODER = "resnet18"
ENCODER_WEIGHT = "imagenet"
NUM_CLASSES = 2
GPU_ID = 0
NUM_ITERATIONS = 5000
VAL_EVERY_IT = 500
PRINT_TRAIN_EVERY_IT = 100
SEMI_SUP_DATA_LIST = '/isi/w/lb27/data/PAG_segmentation/processed/semantic_segmentation/real_data/nital_pag_dataset_noset/perc_val_const/15/train_list.txt'

def get_arg_parser():
    parser = argparse.ArgumentParser(description="UNET: Domain Adaptation Model Training")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="number of output classes (default: 2).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir-image", type=str, required=True,
                        help="Path to the directory containing the images.")
    parser.add_argument("--data-dir-label", type=str, required=False,
                        help="Path to the directory containing the labels.")
    parser.add_argument("--data-list-train", type=str,required=True,
                        help="Path to the file listing the images for training.")
    parser.add_argument("--data-list-validation", type=str,required=False,
                        help="Path to the file listing the images for training.")
    parser.add_argument("--encoder", type=str, default=ENCODER,
                        help="which encoder to use for UNET (default: resnet18)")
    parser.add_argument("--encoder-weights", type=str, default=ENCODER_WEIGHT,
                        help="which encoder weights to use for UNET (default: imagenet)")
    parser.add_argument("--cpu", default=False,action="store_true",
                        help="whether to use cpu for training.")
    parser.add_argument("--gpu-id", default=GPU_ID,
                        help="which gpu to use for training.")
    parser.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS,
                        help=f"number of iterations to train (default: {NUM_ITERATIONS}).")
    parser.add_argument("--val-every-it", type=int, default=VAL_EVERY_IT,
                        help=f"perform validation every nth iteration (default: {VAL_EVERY_IT}).")
    parser.add_argument("--print-train-every-it", type=int, default=PRINT_TRAIN_EVERY_IT,
                        help=f"print losses and metric in training every nth iteration (default: {PRINT_TRAIN_EVERY_IT}).")
    parser.add_argument("--add-clahe",default=False, action='store_true',
                        help=f"add clahe to every image.")
    parser.add_argument("--model-arch",default="UnetPlusPlus",choices=["UnetPlusPlus","Unet","FPN","DeepLabV3"],
                        help=f"which model architecture to use choices are [UnetPlusPlus:default,Unet,FPN,DeepLabV3]")
    
    
    
    
    
    return parser