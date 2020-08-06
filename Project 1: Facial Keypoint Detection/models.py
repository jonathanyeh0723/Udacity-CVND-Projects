## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # 1st conv layer
        self.conv1 = nn.Conv2d(1, 32, 5)
        # output size = (W-F)/S + 1 = (224 - 5)/1 + 1 = 220
        
        #--------------------------------------------------------
        # W — the width/height (square) of the previous layer
        # F — kernel_size
        # S — the stride of the convolution
        #--------------------------------------------------------
        
        # 1st pooling layer
        self.pool1 = nn.MaxPool2d(2, 2)
        # 220/2 = 110, the output Tensor for one image, will have the dimensions: (32, 110, 110)
        
        # 2nd conv layer
        self.conv2 = nn.Conv2d(32, 64, 4)
        # output size = (W-F)/S + 1 = (110 - 4)/1 + 1 = 107
     
        # 2nd pooling layer
        self.pool2 = nn.MaxPool2d(2, 2)
        # 107/2 = 53.5, round down to 53. The output Tensor for one image, will have the dimensions: (64, 53, 53)
        
        # 3rd conv layer
        self.conv3 = nn.Conv2d(64, 128, 3)
        # output size = (W-F)/S + 1 = (53 - 3)/1 + 1 = 51
     
        # 3rd pooling layer
        self.pool3 = nn.MaxPool2d(2, 2)
        # 51/2 = 25.5, round down to 25. The output Tensor for one image, will have the dimensions: (128, 25, 25)
        
        # 4th conv layer
        self.conv4 = nn.Conv2d(128, 256, 2)
        # output size = (W-F)/S + 1 = (25 - 2)/1 + 1 = 24
     
        # 4th pooling layer
        self.pool4 = nn.MaxPool2d(2, 2)
        # 24/2 = 12 the output Tensor for one image, will have the dimensions: (256, 12, 12)  
        
        
        # 256 outputs * the 12*12 filtered/pooled map size
        # Fully conncted layer 1
        self.fc1 = nn.Linear(256*12*12, 1000)   
        # Fully conncted layer 2
        self.fc2 = nn.Linear(1000, 1000)
        
        # finally, create 136 output channels (for the 136 classes)
        self.fc3 = nn.Linear(1000, 136)
        
         #---------------------------------------------------------------------------------------------------------------------
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        drop1 = nn.Dropout(0.1)
        drop2 = nn.Dropout(0.2)
        drop3 = nn.Dropout(0.3)
        drop4 = nn.Dropout(0.4)
        drop5 = nn.Dropout(0.5)
        drop6 = nn.Dropout(0.5)
        
        x = drop1(self.pool1(F.relu(self.conv1(x))))
        x = drop2(self.pool2(F.relu(self.conv2(x))))
        x = drop3(self.pool3(F.relu(self.conv3(x))))
        x = drop4(self.pool4(F.relu(self.conv4(x))))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        x = drop5(F.relu(self.fc1(x)))
        x = drop6(F.relu(self.fc2(x)))
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
        