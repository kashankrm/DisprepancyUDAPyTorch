
import albumentations as A
import albumentations.augmentations.functional as F
import cv2

class ConvertToRGB(A.augmentations.transforms.ImageOnlyTransform):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """



    """Convert the input grayscale image to RGB.
    Args:
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=True, p=1.0):
        super(ConvertToRGB, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        if F.is_rgb_image(img):
            
            return img
        if not F.is_grayscale_image(img):
            raise TypeError("ToRGB transformation expects 2-dim images or 3-dim with the last dimension equal to 1.")

        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def get_transform_init_args_names(self):
        return ()

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
                                A.augmentations.transforms.ToFloat(always_apply=True),
                                A.augmentations.transforms.ToGray(always_apply=True),
                                ConvertToRGB(always_apply=True),
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
        A.augmentations.transforms.ToGray(always_apply=True),
        ConvertToRGB(always_apply=True),
        A.augmentations.transforms.ToFloat(always_apply=True),
        A.Resize(288,288)
    ])
    return aug
def bjoern_augmentation():
    '''
    rotation (65°), shift (15%), shear (10%), zoom (20%) as well as vertical and horizontal flip
    '''
    aug = A.Compose([
                                A.augmentations.transforms.ToFloat(always_apply=True),
                                A.augmentations.transforms.ToGray(always_apply=True),
                                ConvertToRGB(always_apply=True),
                                A.Resize(288,288),
                                A.augmentations.Rotate(limit=65), 
                                A.augmentations.geometric.transforms.Affine(mode=cv2.BORDER_REFLECT101,translate_percent=0.15,shear=0.1,scale=0.2),
                                A.HorizontalFlip(),
                                A.VerticalFlip()
    ])
    return aug


def clahe_augment(aug):
    ret = A.Compose([
        A.CLAHE(always_apply=True),
        aug,
       
    ])
    return ret

def get_old_augmentation():
    '''
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
    
    '''
    train_transform = [
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.augmentations.transforms.ToFloat(always_apply=True),
        A.Resize(288,288),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.RandomCrop(height=250, width=250, always_apply=True),
        A.Resize(288,288),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)

def get_augmentations(aug_type):
    if   aug_type == "clahe+adaptsegnet":
        return clahe_augment(get_basic_train_augment()), clahe_augment(get_basic_val_augment())
    elif aug_type == "clahe+old_repo":
        return clahe_augment(get_old_augmentation()), clahe_augment(get_basic_val_augment())    
    elif aug_type == "adaptsegnet":
        return get_basic_train_augment(), get_basic_val_augment()
    elif aug_type == "old_repo":
        return get_old_augmentation(), get_basic_train_augment()