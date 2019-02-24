# AMLab/AMC Version
The implemented U-net is based on the U-net used for Kaggle's Carvana Image masking Challenge (as seen on,
https://github.com/milesial/Pytorch-UNet),and https://github.com/jvanvugt/pytorch-unet that provides a general framework for U-nets, 
and https://github.com/shreyaspadhy/UNet-Zoo, a collection of UNet and hybrid architectures for the BraTS Brain Tumor Segmentation Challenge.

The U-nets were adapted for our purposes. Nevertheless, closely following the paper introducing U-nets [U-Net](https://arxiv.org/pdf/1505.04597.pdf).

## Usage

### Prediction

To see all options:
`python predict.py -h`

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

You can use the gpu-only version with `-g`.

You can specify which model file to use with `--model MODEL.pth`.

### Training

`python train.py -h` should get you started. 

## Dependencies
This package depends on [pydensecrf](https://github.com/lucasb-eyer/pydensecrf), available via `pip install`.
