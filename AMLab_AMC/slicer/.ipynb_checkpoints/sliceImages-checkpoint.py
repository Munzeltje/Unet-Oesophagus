import numpy as np
from ImageSlicer import ImageSlicer
import re
import os
from PIL import Image
from argparse import ArgumentParser


def slice_images(rootdir,targetdir,mask,maskdir,extension,dim,strides,padding=False):
    """
    input: - rootdir (str): root directory where the subfolders with the source images reside
           - targetdir (str): target directory where the subimages should be saved, note that
                        the folder must already exist and be empty before execution
           - mask (str): mask file name we want to slice corresponding to the image
           - maskdir (str): target directory to save the sliced masks
           - extension (str): image extension; jpg,jpeg,png,tiff.
           - dim ((int,int)): desired subimage dimension
           - strides [int,int]: image overlap stride
    output: folder with all the sliced images from the root directory.
    """
    for directory in os.listdir(rootdir):
        dir_path=rootdir+'/'+directory
        for file in os.listdir(dir_path):
            if extension in file:
                # Source Images
                source_path = dir_path+'/'+file
                slice_image = ImageSlicer(source_path, dim,strides=strides,labels=False,PADDING=padding)
                transformed_image = slice_image.transform()
                slice_image.save_images_directory(transformed_image,targetdir)
                # Masks
                mask_path = dir_path+'/'+mask
                slice_mask = ImageSlicer(mask_path,dim,strides=strides,labels=True,PADDING=padding)
                transormed_mask = slice_mask.transform()
                slice_mask.save_images_directory(transormed_mask,maskdir)
                
                
            


def remove_images(images_path, labels_path,target_dim=256):
    
    for file in os.listdir(images_path):
        image_path = images_path+"/"+file
        label_path = labels_path+"/"+file
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).T; image_shape = image.shape;
        mean = np.mean(image.reshape((image_shape[0],image_shape[1]*image_shape[2])),1)
        if mean[1] > 213 or (not (image_shape[1]==target_dim) or not (image_shape[2]==target_dim)):
            os.remove(image_path)
            os.remove(label_path)
            
    for file in os.listdir(labels_path):
        image_path = images_path+"/"+file
        label_path = labels_path+"/"+file
        image = Image.open(label_path).convert("L")
        image = np.array(image).T
        if not (image.shape[0]==target_dim) or not (image.shape[1]==target_dim):
            os.remove(image_path)
            os.remove(label_path)
        
        
                
                
parser = ArgumentParser()
parser.add_argument("-d", "--dimension", dest="dim",
                    help="Specify dimension to slice the images", action="store", default=256)
parser.add_argument("-s", "--stride",
                    action="store", dest="stride", default=None,
                    help="Specify the stride to overlap")
parser.add_argument("-m", "--mask",
                    action="store", dest="mask", default="mask7.png",
                    help="The mask used for the images")


img_source = "/home/argos/Projects/AMC/Implementation/Unet-Medical/pytorch-unet-master/data/new_train2"
mask_source = "/home/argos/Projects/AMC/Implementation/Unet-Medical/pytorch-unet-master/data/new_train_masks2"

args = parser.parse_args()
dim, stride, mask = int(args.dim), args.stride, args.mask
if not stride==None:  stride = int(stride)

print("Slicing the Source Images...\n")
slice_images("/home/argos/Projects/AMC/UvA_Master/BigTiff/20x",img_source,mask,mask_source,".tiff",(dim,dim),[stride,stride],padding=False)
print("[=======] Done [=======]\n")

print("Removing the unnecessary images...\n")
remove_images(img_source,mask_source,dim)

print("DONE")