import numpy as np
import glob2 as glob
import PIL.Image as Image
import random
import h5py
import matplotlib.pyplot as plt
import os

import torch
import torchvision.transforms.functional as TF
import cv2
import pickle


"""
ISIC Dataset
Class only contains the dataset consisting of those ISIC images with exactly 3 annoations.
Note that there are images with less and more annotations.
"""
# Folder Structure (Files are >50GB and are therefore stored in scratch directory on theia!)
# datapath = /scratch/kmze/isic
# --- Images
# --- Segmentations
class LIDC(torch.utils.data.Dataset):
    images = []
    labels = []
    series_uid = []
    #def __init__(self, transform, apply_symmetric_transforms, data_path):
    def __init__(self, transform, apply_symmetric_transforms, data_path):
        self.images=[]
        self.labels=[]
        self.series_uid=[]
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(data_path):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = data_path + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)

        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def symmetric_augmentation(self, images_and_masks=[]):
        # Random Horizontal Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [TF.affine(x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0)
                            for x in images_and_masks]

        return images_and_masks

    def __getitem__(self, index):
        #image = np.expand_dims(self.images[index], axis=0)
        image = self.images[index]
        #Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0,3)].astype(float)
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        #image = torch.from_numpy(image)
        #label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label, [label], torch.as_tensor([0])

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)


class LIDC2(torch.utils.data.Dataset):
    images = []
    labels = []
    series_uid = []
    #def __init__(self, transform, apply_symmetric_transforms, data_path):
    def __init__(self, transform, apply_symmetric_transforms, data_path):
        self.images=[]
        self.labels=[]
        self.series_uid=[]
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(data_path):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = data_path + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)

        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def symmetric_augmentation(self, images_and_masks=[]):
        # Random Horizontal Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [TF.affine(x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0)
                            for x in images_and_masks]

        return images_and_masks

    def __getitem__(self, index):
        #image = np.expand_dims(self.images[index], axis=0)
        image = self.images[index]
        #Randomly select one of the four labels for this image
        #label = self.labels[index][random.randint(0,3)].astype(float)
        label1 = self.labels[index][0].astype(float)
        label2 = self.labels[index][1].astype(float)
        label3 = self.labels[index][2].astype(float)
        label4 = self.labels[index][3].astype(float)


        if self.transform is not None:
            image = self.transform(image)
            label1 = self.transform(label1)
            label2 = self.transform(label2)
            label3 = self.transform(label3)
            label4 = self.transform(label4)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        #image = torch.from_numpy(image)
        #label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label1 = label1.type(torch.FloatTensor)
        label2 = label2.type(torch.FloatTensor)
        label3 = label3.type(torch.FloatTensor)
        label4 = label4.type(torch.FloatTensor)

        label = []
        label.append(label1)
        label.append(label2)
        label.append(label3)
        label.append(label4)

        return image, label, [label], torch.as_tensor([0])

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)





'''
class LIDC(torch.utils.data.Dataset):
    def __init__(self, transform, apply_symmetric_transforms, data_path):
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms
        data = h5py.File(data_path , 'r')
        train=data['train']
        val=data['val']
        test=data['test']
        images=[]
        anno_1=[]
        anno_2=[]
        anno_3=[]
        anno_4=[]
        tot_anno=[]
        for i in range(train["images"].shape[0]):
            tr=np.array(train["images"][i])
            tr=Image.fromarray(tr)
            #tr=cv2.normalize(np.array(tr), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            #tr=np.array(tr).transpose(1,2,0)
            images.append(tr)
            images.append(tr)
            images.append(tr)
            images.append(tr)
        for i in range(train["labels"].shape[0]):
            tr=np.array(train["labels"][i,...,0])
            anno_1.append(tr)
            anno_1.append(tr)
            anno_1.append(tr)
            anno_1.append(tr)
            tot_anno.append(tr)
            tr=np.array(train["labels"][i,...,1])
            anno_2.append(tr)
            anno_2.append(tr)
            anno_2.append(tr)
            anno_2.append(tr)
            tot_anno.append(tr)
            tr=np.array(train["labels"][i,...,2])
            anno_3.append(tr)
            anno_3.append(tr)
            anno_3.append(tr)
            anno_3.append(tr)
            tot_anno.append(tr)
            tr=np.array(train["labels"][i,...,3])
            anno_4.append(tr)
            anno_4.append(tr)
            anno_4.append(tr)
            anno_4.append(tr)
            tot_anno.append(tr)

        for i in range(val["images"].shape[0]):
            tr=np.array(val["images"][i])
            tr=Image.fromarray(tr)
            #tr=cv2.normalize(np.array(tr), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            #tr=np.array(tr).transpose(1,2,0)
            images.append(tr)
            images.append(tr)
            images.append(tr)
            images.append(tr)
        for i in range(val["labels"].shape[0]):
            tr=np.array(val["labels"][i,...,0])
            anno_1.append(tr)
            anno_1.append(tr)
            anno_1.append(tr)
            anno_1.append(tr)
            tot_anno.append(tr)
            tr=np.array(val["labels"][i,...,1])
            anno_2.append(tr)
            anno_2.append(tr)
            anno_2.append(tr)
            anno_2.append(tr)
            tot_anno.append(tr)
            tr=np.array(val["labels"][i,...,2])
            anno_3.append(tr)
            anno_3.append(tr)
            anno_3.append(tr)
            anno_3.append(tr)
            tot_anno.append(tr)
            tr=np.array(val["labels"][i,...,3])
            anno_4.append(tr)
            anno_4.append(tr)
            anno_4.append(tr)
            anno_4.append(tr)
            tot_anno.append(tr)

        for i in range(test["images"].shape[0]):
            tr=np.array(test["images"][i])
            tr=Image.fromarray(tr)
            #tr=cv2.normalize(np.array(tr), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            #tr=np.array(tr).transpose(1,2,0)
            images.append(tr)
            images.append(tr)
            images.append(tr)
            images.append(tr)
        for i in range(test["labels"].shape[0]):
            tr=np.array(test["labels"][i,...,0])
            anno_1.append(tr)
            anno_1.append(tr)
            anno_1.append(tr)
            anno_1.append(tr)
            tot_anno.append(tr)
            tr=np.array(test["labels"][i,...,1])
            anno_2.append(tr)
            anno_2.append(tr)
            anno_2.append(tr)
            anno_2.append(tr)
            tot_anno.append(tr)
            tr=np.array(test["labels"][i,...,2])
            anno_3.append(tr)
            anno_3.append(tr)
            anno_3.append(tr)
            anno_3.append(tr)
            tot_anno.append(tr)
            tr=np.array(test["labels"][i,...,3])
            anno_4.append(tr)
            anno_4.append(tr)
            anno_4.append(tr)
            anno_4.append(tr)
            tot_anno.append(tr)

        self.images=images
        self.anno_1=anno_1
        self.anno_2=anno_2
        self.anno_3=anno_3
        self.anno_4=anno_4
        self.all_anno=tot_anno

    def symmetric_augmentation(self, images_and_masks=[]):
        # Random Horizontal Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [TF.affine(x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0)
                            for x in images_and_masks]

        return images_and_masks

    def __len__(self):
        # Returns the total number of samples in the DataSet
        return len(self.images)

    def __getitem__(self, idx):
        # Generates one sample of data
        image_path = self.images[idx]
        target1_path = self.anno_1[idx]
        target2_path = self.anno_2[idx]
        target3_path = self.anno_3[idx]
        target4_path = self.anno_4[idx]
        target_path = self.all_anno[idx]

        path_list = [image_path, target_path, target1_path, target2_path, target3_path,
                     target4_path]

        # Prepares images
        def pipe(x):
            #x = x.astype('int16')
            x = self.transform(x)
            x = x.float()
            return x

        transformed_list = [pipe(x) for x in path_list]
        image, target, target1, target2, target3, target4 = transformed_list
        # Activate symmetric transformations for Data Augmentation, see method description of symmetric_augmentation
        if self.symmetric_transforms:
            image, target, target1, target2, target3, target4 = self.symmetric_augmentation(
                transformed_list)

        # Normalize each patch by its mean and variance and set intensities to 0 that are more then 3 stds away
        #mean = torch.mean(image)
        #std = torch.std(image)
        #image = (image-mean)/(3*std)
       # image = TF.normalize(image, torch.mean(image), torch.std(image))
        return image, target, [target1, target2, target3, target4], torch.as_tensor([0])
'''
