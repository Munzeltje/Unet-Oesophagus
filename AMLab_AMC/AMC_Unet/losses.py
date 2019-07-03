# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLossMultiClass(nn.Module):

    def __init__(self):
        super(DiceLossMultiClass, self).__init__()

    def forward(self, output, mask, evaluate=False):
        num_classes = output.size(1)
        dice_eso = 0
        #if evaluate:
        output = torch.softmax(output,dim=1)
        if evaluate:
            output = (output > (1/3)).float()
        for i in range(num_classes):
            probs = torch.squeeze(output[:, i, :, :], 1)
            masks = torch.squeeze(mask[:, i, :, :],1)
            eps = 1e-6
            inter = torch.dot(probs.contiguous().view(-1), masks.contiguous().view(-1))
            union = torch.sum(probs) + torch.sum(masks) + eps
            t = (2 * inter.float() + eps) / union.float()
            dice_eso += t
        
        loss = None
        if not evaluate:
            #print(dice_eso.size(0))
            loss = 1 - (dice_eso / 3)
        else:
            loss = dice_eso / 3
        return loss

class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):

        probs = F.sigmoid(torch.squeeze(output, 1))
        mask = torch.squeeze(mask, 1)
        intersection = probs * mask
        intersection = torch.sum(intersection, 2)

        intersection = torch.sum(intersection, 1)
        den1 = probs * probs
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        den2 = mask * mask
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        eps = 1
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))


        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)

        return loss

class FocalLoss(nn.Module):

    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, output, mask):
        output = output.squeeze(1)
        assert output.shape == mask.shape
        max_val = (-output).clamp(min=0)
        loss = output - output * mask + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()
        invprobs = F.logsigmoid(-output * (mask*2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

class FocalDice(nn.Module):

    def __init__(self, gamma, alpha):
        super(FocalDice, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.dice = DICELoss()
        
    def forward(self, output, mask):
        # We need to add one to the dice since this one computes (1 - dice)
        loss = self.alpha * self.focal(output, mask) - torch.log(-self.dice(output, mask)+1)
        return loss
        
