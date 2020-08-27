# import required packages and files

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import helper
import argparse

# set input options for user 

parser = argparse.ArgumentParser()

parser.add_argument("path_image", help="Please enter the path of your picture",default="flowers/test/11/image_03098.jpg", type=str)
parser.add_argument("checkpoint", help="Please enter the directory you saved your trained model at.", default="checkpoint1.pth", type=str) 
parser.add_argument("--top_k", help="How many of the most likely classes do you want the program to display for the given image? ", default=3, type=int)
parser.add_argument("--category_names", help="Please enter the file name to use a mapping of categories to real names", default="cat_to_name.json", type=str)
parser.add_argument("--gpu", help="Do you want to use gpu? 1 yes 0 no", default=1, type=int) 


args = parser.parse_args()

# load saved model
model = helper.load_checkpoint(args.checkpoint)


# work on gpu/cpu
flag = args.gpu==1
if flag is True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Preprocess image and predict the flower shown    
topk_prob, topk_classes = helper.predict(args.path_image, model, args.top_k, device)

# Label mapping 
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# convert classes to names    
topk_names = []
for k in topk_classes:
        topk_names.append(cat_to_name[k])
        
print(topk_prob, topk_names)

    

                

