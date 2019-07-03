#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
import random
from PIL import Image
from Augmentor.Operations import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
Image.MAX_IMAGE_PIXELS = None


class DataLoad(object):
    """
    Defines the data loader and data operations.
    """
    def __init__(self, dir_img=None, dir_mask=None, dir_val_img=None, dir_val_mask=None,
                 train_and_val=True, val=False):
        if train_and_val:
            self.dir_img = dir_img
            self.dir_mask = dir_mask
            self.dir_val_img = dir_val_img
            self.dir_val_mask = dir_val_mask
            # Get the identities of the images
            ids = self.get_ids(dir_img)
            ids = self.split_ids(ids)
            ids_val = self.get_ids(dir_val_img)
            ids_val = self.split_ids(ids_val)
            # Prepare the data loaders for usage
            self.idtrainset = self.split_train_val(ids)
            self.idvalset = self.split_train_val(ids_val, shuffle=False)
            self.N_train = len(self.idtrainset)
            self.N_val = len(self.idvalset)
        elif val:
            self.dir_val_img = dir_val_img
            self.dir_val_mask = dir_val_mask
            # Get the identities of the images
            ids_val = self.get_ids(dir_val_img)
            ids_val = self.split_ids(ids_val)
            # Prepare the data loaders for usage
            self.idvalset = self.split_train_val(ids_val, shuffle=False)
            self.N_val = len(self.idvalset)

    def get_ids(self, dir):
        """Returns a list of the ids in the directory"""
        return (f[:-4] for f in os.listdir(dir))

    def split_ids(self, ids, n=1):
        """Split each id in n, creating n tuples (id, k) for each id"""
        return ((id, i) for id in ids for i in range(n))


    def yield_imgs(self, ids, dir, suffix, epoch, real_data=True,data_augment=True):
        """
        From a list of tuples, returns the correct (data augmented) image.
        """
        if real_data:
            for id, pos in ids:
                im = np.array(Image.open(dir + id + suffix),dtype=np.float32) 
                yield im

        # Yield the Data augmented images
        if data_augment:
            #print("Using the Data Augmented Images")
            for id, pos in ids:
                #print(id)
                int_id = int(id)+epoch
                random.seed(int_id)
                np.random.seed(int_id)
                im = Image.open(dir + id + suffix)
                transformed = self.data_augmentation(im)
                yield transformed

    def data_augmentation(self, img):
        """
        Data augmentation pipeline, we could experiment
        with colour and the distortions. 
        At the moment, we are not performing colour
        augmentation.
        """
        #rnd_col = RandomColor(8,0.1,1.9)
        rotate = Rotate(0.6,-1)
        #rotaterange = RotateRange(0.8,24,24)
        flip = Flip(0.8,"RANDOM")
        #distort = Distort(0.3,30,50,2)
        transformed = rotate.perform_operation([img])
        transformed = flip.perform_operation(transformed)
        #transformed = rotate.perform_operation(transformed)
        #transformed = rnd_col.perform_operation(transformed)
        #transformed = distort.perform_operation(transformed)
        transformed = np.array(transformed[0],dtype=np.float32)
        return transformed


    def get_imgs_and_masks(self, ids_type,number_imgs,epoch,real_data=True, data_augment=True):
        """Return all the couples (img, mask)"""
        ids = None; dir_img = None; dir_mask = None;
        if ids_type=="train":
            ids = self.idtrainset[:number_imgs]
            dir_img = self.dir_img
            dir_mask = self.dir_mask
        else:
            ids = self.idvalset[:number_imgs]
            dir_img = self.dir_val_img
            dir_mask = self.dir_val_mask
            
        imgs = self.yield_imgs(ids, dir_img,'.png',epoch,real_data,data_augment)
        # need to transform from HWC to CHW
        imgs_switched = map(self.hwc_to_chw, imgs)
        imgs_normalized = map(self.normalize, imgs_switched)
        #imgs_normalized = map(lambda x: x / 255, imgs_switched)
        
        masks = self.yield_imgs(ids, dir_mask,'.png',epoch,real_data, data_augment)
        return zip(imgs_normalized, masks)

    def split_train_val(self, dataset, shuffle=True):
        dataset = list(dataset)
        if shuffle: random.shuffle(dataset)
        return dataset


    def batch(self, iterable, batch_size):
        """Yields lists by batch"""
        b = []
        for i, t in enumerate(iterable):
            b.append(t)
            if (i + 1) % batch_size == 0:
                yield b
                b = []
        if len(b) > 0:
            yield b

    def get_full_img_and_mask(self, id, dir_img, dir_mask):
        im = Image.open(dir_img + id + '.png')
        mask = Image.open(dir_mask + id + '.png')
        return np.array(im), np.array(mask)

    def hwc_to_chw(self, img):
        return np.transpose(img, axes=[2, 0, 1])

    def resize_and_crop(self, pilimg, scale=0.5, final_height=None):
        w = pilimg.size[0]
        h = pilimg.size[1]
        newW = 224
        newH = 224
        if not final_height:
            diff = 0
        else:
            diff = newH - final_height

        img = pilimg.resize((newW, newH))
        img = img.crop((0, diff // 2, newW, newH - diff // 2))
        return np.array(img, dtype=np.float32)

    def normalize(self, x, unit_var=False):
        R, G, B = x[0,:,:], x[1,:,:], x[2,:,:]

        R_mean, G_mean, B_mean = R.mean(), G.mean(), B.mean()
        if unit_var:
            R_std, G_std, B_std = R.std(), G.std(), B.std()
            R = self.unit_variance(R,R_mean, R_std)
            G = self.unit_variance(G,G_mean, G_std)
            B = self.unit_variance(B,B_mean, B_std)
        else:
            R = (R - R_mean) / 255; G = (G - G_mean) / 255; B = (B - B_mean) / 255;
        x[0,:,:] = R; x[1,:,:] = G; x[2,:,:] = B;
        #print(x)
        #raise ValueError("hi")
        return x 
    
    def unit_variance(self, x, mean, std):
        return (x - mean) / std

    # credits to https://stackoverflow.com/users/6076729/manuel-lagunas
    def rle_encode(self, mask_image):
        pixels = mask_image.flatten()
        # We avoid issues with '1' at the start or end (at the corners of
        # the original image) by setting those pixels to '0' explicitly.
        # We do not expect these to be non-zero for an accurate mask,
        # so this should not harm the score.
        pixels[0] = 0
        pixels[-1] = 0
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
        runs[1::2] = runs[1::2] - runs[:-1:2]
        return runs
    
    def plot_img_and_mask(self, img, mask):
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        a.set_title('Input image')
        plt.imshow(img)

        b = fig.add_subplot(1, 2, 2)
        b.set_title('Output mask')
        plt.imshow(mask)
        plt.show()
