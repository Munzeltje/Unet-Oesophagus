import sys
import os
from optparse import OptionParser
import numpy as np
# Torch dependencies
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
# Our own written dependencies
from metrics import Metric
from Unet import UNet
from Transfer import *
from losses import DICELoss, FocalLoss, FocalDice
from load import DataLoad
# The new Pytorch update has tensorboard imbedded
# consider it
from torchvision import models
from tensorboardX import SummaryWriter

def predict_img(net, img, mask, device):
    # Get the images and true_masks from the batches and prepare for use
    img = torch.tensor(img).unsqueeze(0).to(device)
    mask = torch.tensor(mask).unsqueeze(0).to(device)
    mask = (mask > 0).float()
    # Pass them trough the U-net model and get a prediction
    prediction = torch.sigmoid(net(img))
    return prediction, mask


def predict_images(net, dataloader):
    net.eval()
    dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_val = int(Data_Loader.N_val)
    #N_val = 10
    print("Number of total images reserved for validation: ", N_val, "Using Device: ", device)
    metric = Metric()
    validate = dataloader.get_imgs_and_masks("val",N_val, 0, real_data=True, data_augment=False)
    preds, masks = torch.zeros(N_val, dim**2), torch.zeros(N_val, dim**2)
    #print(preds.shape)
    print(metric.evaluate(net, validate))
    raise ValueError("hi")
    with torch.no_grad():
        for i, batch in enumerate(validate):
            img, true_mask = batch
            predictions, true_mask = predict_img(net,img,true_mask, device)
            preds[i,:] = predictions.view(-1)
            masks[i,:] = true_mask.view(-1)
    preds, masks = preds.view(-1), masks.view(-1)
    roc_auc = metric.plot_roc(preds, masks)
    #print(roc_auc)
            
def get_roc(metric, preds, masks):
    return metric.plot_roc(preds, masks)

# Define the terminal arguments
def config():
    parser = OptionParser()

    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-d', '--dim', dest='dim', default=256, help='dimensionality of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    # Get the argumetns given through the terminal
    config = config()
    net_dense161 = DenseNet161(num_classes=1,num_filters=32,pretrained=True, is_deconv=False,freeze=False)
    # If we would like to load a trained model
    #net_dense161.load_state_dict(torch.load("checkpoints/First_Models/Dense_five_ep10_07550.pth"))
    #net_dense161.load_state_dict(torch.load("checkpoints/First_Models/Dense_five_512_ep10_0750.pth"))
    path = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Code/Slide_classification/models/Dense_seven_test.pth"
    net_dense161.load_state_dict(torch.load(path))

    print('Model loaded: {}'.format(config.load))
    
    if config.gpu: 
        net_dense161.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory
    
    #dir_test_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/five/test/"
    #dir_test_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/five/test_masks/"
    #dir_test_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/mc_512/Pos/strictly_pos/five/test/"
    #dir_test_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/mc_512/Pos/strictly_pos/five/test_masks/"
    dir_test_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/patched_wsi/seven/patched/"
    dir_test_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/patched_wsi/seven/masks/"
    Data_Loader = DataLoad(None, None, dir_test_img, dir_test_mask, val=True, train_and_val=False)
    predict_images(net_dense161, Data_Loader)

    print("Done Validating")

