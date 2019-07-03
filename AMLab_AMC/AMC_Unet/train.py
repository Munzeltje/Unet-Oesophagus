import sys
import os
from optparse import OptionParser
import numpy as np
from PIL import Image
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
#torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#np.random.seed(0)

def train_net(net,epochs=12,batch_size=32,lr=5e-5,validate=True,save_cp=True):
    
    #dir_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Pos/train_pos/"
    #dir_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Pos/train_masks_five_seven_pos/"
    #dir_test_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Pos/test_pos/"
    #dir_test_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Pos/test_masks_five_seven_pos/"
    
    #dir_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/five/train/"
    #dir_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/five/train_masks/"
    #dir_test_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/five/test/"
    #dir_test_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/five/test_masks/"
    
    dir_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/seven/train/"
    dir_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/seven/train_masks/"
    dir_test_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/seven/test/"
    dir_test_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Pos/final_pos/seven/test_masks/"
    
    #dir_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/mc_512/Pos/strictly_pos/five/train/"
    #dir_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/mc_512/Pos/strictly_pos/five/train_masks/"
    #dir_test_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/mc_512/Pos/strictly_pos/five/test/"
    #dir_test_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/mc_512/Pos/strictly_pos/five/test_masks/"
    
    #dir_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Non_white/seven/train/"
    #dir_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Non_white/seven/train_masks/"
    #dir_test_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Non_white/seven/test/"
    #dir_test_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Non_white/seven/test_masks/"
    
    #dir_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Non_white/seven/less_black/train/"
    #dir_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Non_white/seven/less_black/train_masks/"
    #ir_test_img = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Non_white/seven/less_black/test/"
    #dir_test_mask = "/data/ml/AMC-data/Oesophagus_Unet/3-Maskers/Data/Final/Non_white/seven/less_black/test_masks/"


    # The direcotry where we would like to save our trained models
    dir_checkpoint = 'checkpoints/'
    Data_Loader = DataLoad(dir_img, dir_mask, dir_test_img, dir_test_mask)
    # The device we are using, naturally, the gpu by default
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device: ",device)
    
    # The number of data points we're training on
    N_train = Data_Loader.N_train
    N_val = Data_Loader.N_val
    
    # Log the loss and others
    #writer = SummaryWriter('runs/Metric')
    
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr,N_train,
               N_val, str(save_cp), device))
    
    # The optimizer, optim.SGD works just fine but Adam converges faster
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay = 5e-4)
    #optimizer = optim.Adamax(net.parameters(),lr=lr,weight_decay = 6e-4)
    #optimizer = optim.SGD(net.parameters(), lr=lr,momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma = 0.90)
    #weight_decay = 1e-5

    # Beware that the sigmoid/softmax is computed implicitly in BCEloss
    # Therefore, we do not need to apply the sigmoid/softmax function
    # ourselves to the logits
    #criterion = nn.BCEWithLogitsLoss()
    criterion = DICELoss()
    #criterion = FocalDice(gamma=0.25, alpha=0.25)
    iterations = 0
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # Tell pytorch we are in training mode
        net.train()
        # handle the data loading; reset the data generators
        train = Data_Loader.get_imgs_and_masks("train",N_train, epoch,real_data=False,data_augment=True)
        # Keep track of the training loss, we need to save this later on
        epoch_loss = 0; t = 0;
        
        # Training using the specified batch size
        for i, batch in enumerate(Data_Loader.batch(train, batch_size)):
            # Get the images and true_masks from the batches and prepare for use
            imgs = torch.tensor([b[0] for b in batch]).to(device)
            true_masks = torch.tensor([b[1] for b in batch]).to(device)

            true_masks = (true_masks > 0).float()
            # Pass them trough the U-net model and get a prediction
            masks_probs = net(imgs)
            loss = criterion(masks_probs,true_masks)
            # update the loss
            epoch_loss += loss.item()
            # perform the backward pass and update the weights
            loss.backward()
            optimizer.step()
            # flush the gradients
            optimizer.zero_grad()
            
            # print, validate and save
            iterations += 1; t += 1
            #print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train,epoch_loss / (t+1)))   
            if i % 200 == 0 and not i == 0:
                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train,epoch_loss / (t+1)))   
                if validate: 
                    train_test = Data_Loader.get_imgs_and_masks("train",5000,epoch,real_data=False,data_augment=True)
                    # validate on half of the test set, keep the other half for testing
                    val = Data_Loader.get_imgs_and_masks("val",1500,epoch,real_data=True,data_augment=False)
                    test_net(net, iterations, train_test, val, epoch=epoch)
                # Set the model to train mode again
                net.train()
                #save_model(net,dir_checkpoint+'CP{}.pth'.format(iterations), save_model=save_cp)
            scheduler.step()
        # test after epoch
        #test_net(net,iterations,trainset,valset,epoch=epoch)
    # Don't save the model for now
    save_model(net,dir_checkpoint + 'Dense{}.pth'.format(iterations), save_model=save_cp)

def save_model(model, save_path, save_model=False):
    if save_model:
        torch.save(model.state_dict(), save_path)
        print('Checkpoint saved in {}!'.format(save_path))

def test_net(net,t,train,val,epoch=1,validate_trainset=False):
    # Log the metrics
    writer = SummaryWriter('runs/Metric')
    print("Starting Validation")
    # The device we are using, naturally, the gpu by default
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = Metric(device=device)
    # Keep track of the training loss, we need to save this later on
    val_dice, val_accuracy = metric.evaluate(net,val,t)
    print('Validation Dice: {0:.4g}  [===] Validation Accuracy: {1:.4g}'.format(val_dice, val_accuracy))

    if validate_trainset:
        train_dice, train_accuracy = metric.evaluate(net,train,t)
        print('Training Dice: {0:.4g} [==] Training Accuracy: {1:.4g}'.format(train_dice, train_accuracy))
        writer.add_scalars('Train_Val_v4/Dice Score', {'Train Dice':train_dice,'Val Dice':val_dice},t)
        writer.add_scalars('Train_Val_v4/Pixel-Wise Accuracy', {'Train Accuracy':train_accuracy,'Val Accuracy':val_accuracy},t)
            
# Define the terminal arguments
def config():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-v', '--validate', dest='validate', 
                  default=False, help='Validation option')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    # Get the argumetns given through the terminal
    config = config()
    net_dense161 = DenseNet161(num_classes=1,num_filters=32,pretrained=True, is_deconv=False,freeze=False)
    #resnet152 = ResNet152(num_classes=1,num_filters=32,pretrained=True, is_deconv=False,freeze=False)
    # If we would like to load a trained model
    if config.load:
        #net.load_state_dict(torch.load("./checkpoints/CP12.pth"))
        print('Model loaded: {}'.format(config.load))
    
    if config.gpu: 
        net_dense161.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory
    train_net(net_dense161, epochs=10, validate=True,batch_size=32,lr=5e-4)
    print("Done Training")

