import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # Covolutional Layers
#        32 - 5 +1 = 28 
        self.conv1 = nn.Conv2d(1,32,5) # W = 224 - 7 +1 = 218, 218*218*32
        self.conv1_bn = nn.BatchNorm2d(32)
    #114*114*32
        self.conv2 = nn.Conv2d(32,64,3) #114 - 5 + 1 = 110, 110*110*64
        self.conv2_bn = nn.BatchNorm2d(64)
        #55*55*32
        self.conv3 = nn.Conv2d(64,128,3)#55-3+1 = 53, 53*53*128
        self.conv3_bn = nn.BatchNorm2d(128)
        #26*26*10
        self.conv4 = nn.Conv2d(128,256,2)# 26 - 1 = 25, 25 * 25 * 256
        self.conv4_bn = nn.BatchNorm2d(256)
        #12*12*256

        # Maxpooling Layer
        self.pool = nn.MaxPool2d(2,2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(12*12*256,1000)
        self.fc1_bn = nn.BatchNorm1d(1000)        
#         self.fc2 = nn.Linear(1000,500)
#         self.fc2_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(1000, 136)
        # Dropouts

        self.drop1 = nn.Dropout(p = 0.3)
        self.drop2 = nn.Dropout(p = 0.5)




    def forward(self, x):

        # First - Convolution + Activation + Pooling 
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        #print("First size: ", x.shape)

        # Second - Convolution + Activation + Pooling 
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        #print("Second size: ", x.shape)

        # Third - Convolution + Activation + Pooling 
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        #print("Third size: ", x.shape)

        # Forth - Convolution + Activation + Pooling 
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        #print("Forth size: ", x.shape)

        # Flattening the layer
        x = x.view(x.size(0), -1)
        #print("Flatten size: ", x.shape)

        # First - Dense + Activation + Dropout
        x = self.drop1(F.relu(self.fc1_bn(self.fc1(x))))
        #print("First dense size: ", x.shape)

        # Final Dense Layer
        #x = self.drop2(F.relu(self.fc2_bn(self.fc2(x))))
        #print("Final dense size: ", x.shape)
        x = self.fc2(x)
        return x