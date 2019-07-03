import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from scipy import interp
import numpy as np
from PIL import Image
import os

from losses import DiceLossMultiClass
from load import DataLoad 
np.set_printoptions(threshold=np.nan)

class Metric(object):
    """
    Various metrics, including the dice coefficient for, for individual examples. 
    This method currently does not take multi-class into account or binary one-hot vectors 
    for that matter. We need to change it as soon as possible.
    """
    def __init__(self, compute_jaccard=False, device="cuda"):
        print("Initiated Metric Evaluation")
        self.compute_jaccard = compute_jaccard
        self.device = device
    
    def dice(self, input, target):
        ''' 
        Given an input and target compute the dice score
        between both. Dice: 1 - (2 * |A U B| / (|A| + |B|))
        '''
        eps = 1e-6
        if len(input.shape) > 1:
            input, target = input.view(-1), target.view(-1)
        else:
            input, target = torch.tensor(input.astype(float)), torch.tensor(target.astype(float))
        inter = torch.dot(input, target)
        union = torch.sum(input) + torch.sum(target) + eps
        dice = (2 * inter.float() + eps) / union.float()
        return dice

    def pixel_wise(self, input, target):
        """
        Regular pixel_wise accuracy metric, we just
        compare the number of positives and divide it
        by the number of pixels.
        """
        # Flatten the matrices to make it comparable
        input = input.view(-1)
        target = target.view(-1)
        correct = torch.sum(input==target)
        return (correct.item() / len(input))
        
    def jaccard(self, prediction, mask):
        """
        Given a one dimension prediction vector and
        one dimensional mask vector, compute the jaccard
        similarity metric using SciKit-Learn, the vectors
        are required to be numpy vectors. 
        
        TODO: Implement torch version
        """
        return jaccard_similarity_score(prediction, mask)
    
    def evaluate(self, net, dataset, t=0):
        """
        Evaluation without the densecrf (prediction.py, we are not using
        that method, eval_net suffices to predict the images and debugging) 
        with the dice coefficient or some other metric.
        """
        # Tell pytorch we are evaluating
        net.eval()
        total_accuracy, total_dice, total_jaccard = 0, 0, 0
        # To make sure we do not get memory errors
        epoch_loss, h = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                img, true_mask = batch
                # send both to the gpu for evaluation
                img = torch.from_numpy(img).unsqueeze(0).to(self.device)
                true_mask = torch.from_numpy(true_mask).unsqueeze(0).to(self.device)
                # Make sure we are using binary values
                true_mask = (true_mask > 0).float()
                # The model returns logits, so we need it to pass through
                # a sigmoid to get the probabilities
                mask_pred = torch.sigmoid(net(img))
                # threshold the predictions to get the hard per pixel classification
                # for visualization, the probabilities will do just fine
                mask_pred_s = (mask_pred > 0.5).float()
                # beware that, for the moment, we need to pass it the 
                # hard per pixel classification
                model_accuracy = self.pixel_wise(mask_pred_s, true_mask)
                dice = self.dice(mask_pred_s,true_mask)
                jaccard = None
                if self.compute_jaccard: 
                    mask_squashed = mask_pred_s.view(-1)
                    true_mask_squashed = true_mask.view(-1)
                    jaccard = self.jaccard(mask_pred_s, true_mask)
                    total_jaccard += jaccard
                    
                total_accuracy += model_accuracy
                total_dice += dice.item()
                print(dice.item())
                self.save_images(img, mask_pred, true_mask, False, t)
                h += 1
                
        if not self.compute_jaccard:
            return total_dice / (h + 1), total_accuracy / (h + 1)
        return total_dice / (h + 1), total_accuracy / (h + 1), total_jaccard / (h + 1)
    
    def compute_accuracy(self, prediction, mask):
        thresholded = (prediction > 0.5).float()
        pixel_accuracy = self.pixel_accuracy(thresholded, mask)
        dice_score = self.dice(thresholded, mask)
        return pixel_accuracy, dice_score
    
    def plot_roc(self, predictions, true_values):
        """
        Computes the ROC Curve and computes the Area Under the Curve (AUC). 
        Also used to estimate the best threshold, given that we need to minimize 
        false negatives.
        Input: predictions of shape (N,)
               true_values of shape (N,)
        """
        fpr, tpr = dict(), dict()
        true_values = true_values.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        thresholded = (predictions > 0.5)
        acc = np.sum(true_values==thresholded) / true_values.shape[0]
        print("accuracy: ", acc)
        dice = self.dice(thresholded, true_values)
        print("dice:",dice)
        fpr, tpr, thresholds = roc_curve(true_values, predictions)
        index_cutoff = np.argmax(tpr - fpr)
        fpr_cutoff, tpr_cutoff = fpr[index_cutoff], tpr[index_cutoff]
        optimal_cutoff = thresholds[np.argmax(tpr - fpr)]
        print("fpr at cutoff: ", fpr_cutoff)
        print("tpr at cutoff: ", tpr_cutoff)
        print("optimal cutoff: ", optimal_cutoff)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkred', lw=2, label="ROC Curve (area = %0.2f, F1 = %0.2f )" % (roc_auc, dice))
        plt.plot([0,1],[0,1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.title("ROC Curve Model Densenet on 512x512 LGD patches")
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig("roc_five_512.png",dpi=300)
        return roc_auc
    
    def one_hot_predictions(self, predictions):
        """
        Given a pytorch vector with probability predictions,
        convert to a one hot representation, assuming a binary instance.
        """
        predictions = predictions.cpu().detach().numpy()
        one_hot = np.zeros((predictions.shape[0],2))
        for i, pred in enumerate(predictions):
            one_hot[i,0] = 1 - pred
            one_hot[i,1] = pred
        return one_hot
            
    
    def save_images(self, tissue, prediction, mask, condition, t):
        ''' 
        for visualization purposes, saves the image of the tissue patch
        in question alongside the predicted patch and the mask for the corresponding
        path.
        '''  
        path= "/home/bcardenasguevara/Unet_Cleaned/AMLab_Unet/data/predictions"
        if condition:
            pred = prediction.squeeze().cpu().detach().numpy()
            tissue = tissue.squeeze().cpu().detach().numpy().T
            mask = mask.squeeze().cpu().detach().numpy()
            plt.imsave(os.path.join(path, str(t)+"pred"),pred)
            plt.imsave(os.path.join(path, str(t)+"cell"),tissue)
            plt.imsave(os.path.join(path, str(t)+"true"),mask, cmap=cm.gray)
            
