# import the required packages

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from PIL import Image

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("data_directory", help="Please enter the directory your data is stored in",default='flowers', type=str)
parser.add_argument("--gpu", help="Do you want to use gpu? 1 yes 0 no", default=1, type=int) 
parser.add_argument("--model", help="Please choose model architecture from: vgg16 and vgg19 ", default="vgg16", type=str)
parser.add_argument("--learning_rate", help="Please enter your desired learning rate for the optimizer", default="0.0001", type=float)
parser.add_argument("--hidden_units", help="Please enter the number of units for the fully connected hidden layer", default="4096", type=int)
parser.add_argument("--nr_epochs", help="Please enter the number of epochs", default="3", type=int)
parser.add_argument("--save_dir", help="Please enter the directory you want your model to be saved at", default="checkpoint1", type=str)




args = parser.parse_args()

data_dir = args.data_directory 

# Define transform for the training-, validation-, and testing data 

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Assign data directories

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Loading the datasets with ImageFolder

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(test_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(valid_dir, transform=test_transforms)


# Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

image_datasets = [train_data, valid_data, test_data]

    
# work on gpu/cpu
flag = args.gpu==1
if flag is True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

# download both model options

model_vgg16 = models.vgg16(pretrained=True)
model_alexnet = models.alexnet(pretrained=True)

#assign the chosen model
if args.model == "vgg16":
    model = model_vgg16
    input = 25088
else:
    model = model_alexnet
    input = 9216


for param in model.parameters():
    param.requires_grad = False
    
# build classifier and append to pretrained network
    
from collections import OrderedDict

nr_hidden_units = args.hidden_units 

model.classifier = nn.Sequential(OrderedDict([
                          ('dropout1', nn.Dropout(p=0.2)),
                          ('fc1', nn.Linear(input, nr_hidden_units)),
                          ('relu1', nn.ReLU()), 
                          ('dropout2', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(nr_hidden_units,1000)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device);

# Train the Network 

epochs = args.nr_epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        steps +=1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(testloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
            
          
# append mapping of classes to indices to model

model.class_to_idx = train_data.class_to_idx

#Save the checkpoint 

model.class_to_idx = train_data.class_to_idx

states = {
         'model' : model,
         'model.classifier': model.classifier,
         'model.class_to_idx' : model.class_to_idx,
         'epochs': epochs,
         'state_dict': model.state_dict(),
         'optimizer' : optimizer,
         'optimizer.state_dict': optimizer.state_dict()
         }


torch.save(states, 'checkpoint1.pth')

                