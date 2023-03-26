import torch
import os
import cv2
import shutil
from torch.utils.data import Dataset

from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import os.path as osp


class CustomDataset(Dataset):
    def __init__(self, dataset_path_images, list_path, dataset_path_labels=None,preprocessing=None, augment=None,scratch_dir=None):

        

        self.dataset_path_images = dataset_path_images
        if dataset_path_labels is None:
            dataset_path_root = osp.sep.join(self.dataset_path_images.split('/')[:-1])
            if osp.exists(osp.join(dataset_path_root, "labels")):
                dataset_path_labels = dataset_path_root+"/labels"
            else:
                dataset_path_labels = dataset_path_root+"/masks"
        self.dataset_path_labels = dataset_path_labels
        if scratch_dir :
            dataset_path_images = self.dataset_path_images
            dataset_path_labels = self.dataset_path_labels
            self.dataset_path_images = f"{scratch_dir}/{dataset_path_images}"
            self.dataset_path_labels = f"{scratch_dir}/{dataset_path_labels}"
            # if os.path.exists(self.dataset_path_images): print(f"deleting {dataset_path_images} -> {scratch_dir}");shutil.rmtree(self.dataset_path_images)
            # if os.path.exists(self.dataset_path_labels): print(f"deleting {dataset_path_labels} -> {scratch_dir}");shutil.rmtree(self.dataset_path_labels)
            if not os.path.exists(self.dataset_path_images): print(f"copying {dataset_path_images} -> {scratch_dir}");shutil.copytree(dataset_path_images,self.dataset_path_images)
            if not os.path.exists(self.dataset_path_labels): print(f"copying {dataset_path_labels} -> {scratch_dir}");shutil.copytree(dataset_path_labels,self.dataset_path_labels)

        self.list_path = list_path

        self.preprocessing = preprocessing
        self.augment = augment
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []

        for name in self.img_ids:
            if name == "":
                continue
            img_file = osp.join(self.dataset_path_images, name)
            label_file = osp.join(self.dataset_path_labels, name )
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
            
        

        
    def augment_albumentation(self, img, segmentation):
        mask = (np.array(segmentation).astype(np.uint8))
        if mask.max() == 1:
            #rest of the code expects foreground to be 255 rather than True or 1
            mask = mask*255
            segmentation = Image.fromarray(mask)
        if  self.augment:
            
            augmented = self.augment(image=np.array(img), mask=np.array(segmentation)) # from pixelated image to matrix of number
            
            img = augmented["image"]
            segmentation = Image.fromarray(augmented["mask"].astype(np.uint8))
        else:
            img = np.array(img)
        # transform to tensor
        segmentation_np = (np.array(segmentation)/255).astype(np.int64)
        if self.preprocessing:
                sample = self.preprocessing(image=img, mask=segmentation_np)
                img, segmentation = sample['image'].transpose(1,2,0), sample['mask']
        img = TF.to_tensor(img)
        
        segmentation = TF.to_tensor(segmentation) # see notebook img_np, get gradient and create weight mask
        segmentation = segmentation.squeeze(0).long()
        
        return img, segmentation
    
    
    def __getitem__(self, index):# get the augmented images
        datafiles = self.files[index]
        img = Image.open(datafiles["img"])

        segmentation = Image.open(datafiles["label"])
        if segmentation.mode != 'L':
            segmentation = segmentation.convert('L')
        
        name = datafiles["name"]
        
        img, segmentation = self.augment_albumentation(img, segmentation)
        
        #duplicate the images to make it 3-channel
        img = img.expand(3, -1, -1)

        size = img.shape

        return img, segmentation, size, name
        

    def __len__(self):
        return len(self.files)
    
