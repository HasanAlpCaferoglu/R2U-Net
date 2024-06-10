import torch
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw
import numpy as np

def get_accuracy(SR, GT, threshold = 0.5):

    SR[SR>threshold] = 1
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2) # size(0) = batch size, size(1) = height, size(2) = width
    accuracy = float(corr)/float(tensor_size)
    return accuracy


def get_sensitivity(SR, GT, threshold=0.5):
    """
    Sensitivity == Recall

    Note that out_val variable is specific for the Oxford PET Dataset 
    """ 
    out_val = 0.0078 # this variable is specific to Oxford PET Dataset
    SR = SR > threshold
    GT = GT != out_val

    TP = ((SR==1)*(GT==1)) == 1
    FN = ((SR==0)*(GT==1)) == 1

    sensitivity = float(torch.sum(TP)) / (float(torch.sum(TP) + torch.sum(FN)) + 1e-6)
    return sensitivity


def get_specificity(SR, GT, threshold=0.5):
    """
    Note that out_val variable is specific for the Oxford PET Dataset 
    """ 

    out_val = 0.0078
    SR = SR > threshold
    GT = GT != out_val

    TN = ((SR==0)*(GT==0)) == 1
    FP = ((SR==1)*(GT==0)) == 1

    specificity = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return specificity


def get_precision(SR, GT, threshold=0.5):
    """
    Note that out_val variable is specific for the Oxford PET Dataset 
    """ 
    out_val = 0.0078
    SR = SR > threshold
    GT = GT != out_val

    TP = ((SR==1)*(GT==1)) == 1
    FP = ((SR==1)*(GT==0)) == 1

    precision = float(torch.sum(TP)) / (float(torch.sum(TP+FP)) + 1e-6)
    return precision

def get_F1(SR, GT, threshold=0.5):
    recall = get_sensitivity(SR, GT, threshold=threshold)
    precision = get_precision(SR, GT, threshold=threshold)

    F1 = 2*recall*precision / (recall + precision + 1e-6)
    return F1

def get_JS(SR, GT, threshold=0.5):
    """
    JS = Jaccard Similarity
    """
    SR[SR > threshold] = 1
    intersection = torch.sum((SR*GT)==1)
    union = torch.sum((SR+GT)>=1)
    JS = intersection / (union + 1e-6)

def get_DC(SR, GT, threshold=0.5):
    """
    DC: Dice Coefficient

    SR.shape = [batch_size, height, width] = [4, 256, 256]
    GT.shape = [batch_size, height, width] = [4, 256, 256]
    """

    intersection = torch.sum(SR * GT)
    union = torch.sum(SR) + torch.sum(GT)
    DC = (2. * intersection) / (union + 1e-6)
    DC = DC.item()
    return DC
