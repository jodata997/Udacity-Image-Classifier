import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import argparse


#loading a checkpoint 

def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    model.classifier = checkpoint['model.classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
    epoch = checkpoint['epochs']
    model.eval()
    
    return model


# Image Preprocessing 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    image_rs = image.resize((255,255))
    le = (255-224)/2
    to = (255-224)/2
    ri = (255+224)/2
    bu = (255+224)/2
    image_c = image_rs.crop((le,to,ri,bu))
    np_image = np.array(image_c)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2,0,1))
    image_final = torch.from_numpy(np_image)
    
    return image_final

# Predict flower for a given image 

def predict(image_path, model, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path)    
    image = image.type(torch.FloatTensor)
    image = image.to(device)
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    
    topk = ps.topk(top_k)
    topk_indices = np.array(topk[1])
    class_to_idx_inv  = dict(zip(model.class_to_idx.values(), model.class_to_idx.keys()))
    
    topk_classes = []
    for k in topk_indices[0]:
        topk_classes.append(class_to_idx_inv[k])
    
    topk_prob = topk[0]
    
    return topk_prob, topk_classes
    

