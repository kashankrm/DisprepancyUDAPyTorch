
import albumentations as A
def get_basic_train_augment():
    kwargs = {"brightness_limit": 0.02,
                    "brightness_prob": 0,
                    "contrast_limit": 0.15,
                    "contrast_prob": 0,
                    "gaussBlur_limit": 9, 
                    "gaussBlur_prob": 0.2, 
                    "motionBlur_limit": 7, 
                    "motionBlur_prob": 0.2, 
                    "flip_prob": 0.25, 
                    "rot90_prob": 0.25, 
                    "shift_limit": 0.1, 
                    "scale_limit": 0.1, 
                    "rot_limit": 30, 
                    "shiftscalerot_prob": 0.8, 
                    "elastic_alpha": 40.523, 
                    "elastic_sigma": 7.7688, 
                    "elastic_prob": 0.35, 
                    "optDist_distort_limit": 0.14482,
                    "optDist_shift_limit": 0.4154, 
                    "optDist_prob": 0.25,
                }
    aug = A.Compose([
                                A.Resize(288,288),
                                A.RandomBrightness(limit=kwargs["brightness_limit"], p=kwargs["brightness_prob"]),
                                A.RandomContrast(limit=kwargs["contrast_limit"], p=kwargs["contrast_prob"]), # randomly changes the contrast to avoid mistakes caused by different contrast from ECM
                                A.GaussianBlur(blur_limit=kwargs["gaussBlur_limit"], p=kwargs["gaussBlur_prob"]), # smoothening of the image 
                                A.MotionBlur(blur_limit=kwargs["motionBlur_limit"], p=kwargs["motionBlur_prob"]), 
                                A.Flip(p=kwargs["flip_prob"]), 
                                A.RandomRotate90(p=kwargs["rot90_prob"]), 
                                A.ShiftScaleRotate(shift_limit=kwargs["shift_limit"], scale_limit=kwargs["scale_limit"], rotate_limit=kwargs["rot_limit"], 
                                                    border_mode=4, p=kwargs["shiftscalerot_prob"]), # border mode could be changed to 2 or do not pass any values
                                A.ElasticTransform(alpha_affine=0, alpha=kwargs["elastic_alpha"], sigma=kwargs["elastic_sigma"], 
                                                    border_mode=4, approximate=True, p=kwargs["elastic_prob"]), # border mode could be changed to 2 or do not pass any values
                                A.OpticalDistortion(distort_limit=kwargs["optDist_distort_limit"], shift_limit=kwargs["optDist_shift_limit"], 
                                                    border_mode=4, p=kwargs["optDist_prob"])]) # border mode could be changed to 2 or do not pass any values
    return aug
def get_basic_val_augment():
    aug = A.Compose([
        A.Resize(288,288)
    ])
    return aug
def bjoern_augmentation():
    '''
    rotation (65°), shift (15%), shear (10%), zoom (20%) as well as vertical and horizontal flip
    '''
    aug = A.Compose([
                                A.augmentations.transforms.ToFloat(always_apply=True),
                                A.Resize(288,288),
                                A.augmentations.Rotate(limit=65), 
                                A.augmentations.geometric.transforms.Affine(mode=cv2.BORDER_REFLECT101,translate_percent=0.15,shear=0.1,scale=0.2),
                                A.HorizontalFlip(),
                                A.VerticalFlip()
    ])
    return aug
