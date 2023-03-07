import math
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper

class BBBmymodel(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBmymodel, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 8, 3, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = BBBConv2d(8, 16, 3, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = BBBConv2d(16, 32, 3, bias=True, priors=self.priors)
        self.act3 = self.act()
        self.pool3 = nn.MaxPool2d(2,2)
    
        self.flatten = nn.Flatten(1)
        
        self.fc1 = BBBLinear(1152, 512, bias=True, priors=self.priors)
        self.act4 = self.act()
        self.fc2 = BBBLinear(512, 32, bias=True, priors=self.priors)
        self.act5 = self.act()
        self.fc3 = BBBLinear(32, outputs, bias=True, priors=self.priors)
        

class BBBmymodel1(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBmymodel1, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        # self.conv1 = BBBConv2d(inputs, 24, 3, bias=True, priors=self.priors)
        # self.act1 = self.act()
        # self.pool1 = nn.MaxPool2d(6,6)
        # self.conv2 = BBBConv2d(24, 48, 3, bias=True, priors=self.priors)
        # self.act2 = self.act()
        # self.pool2 = nn.MaxPool2d(6,6)
    
    
        # self.flatten = nn.Flatten(1)
        
        # self.fc1 = BBBLinear(1920, 1024, bias=True, priors=self.priors)
        # self.act3 = self.act()
        
        # self.fc2 = BBBLinear(1024, outputs, bias=True, priors=self.priors)
        self.conv1 = BBBConv2d(inputs, 8, 3, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = BBBConv2d(8, 16, 3, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.conv3 = BBBConv2d(16, 32, 3, bias=True, priors=self.priors)
        self.act3 = self.act()
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = BBBConv2d(32, 64, 3, bias=True, priors=self.priors)
        self.act4 = self.act()
        self.pool4 = nn.MaxPool2d(2,2)
        self.conv5 = BBBConv2d(64, 128, 3, bias=True, priors=self.priors)
        self.act5 = self.act()
        self.pool5 = nn.MaxPool2d(2,2)

    
    
        self.flatten = nn.Flatten(1)
        
        self.fc1 = BBBLinear(4096, 1024, bias=True, priors=self.priors)
        self.act6 = self.act()
        self.fc2 = BBBLinear(1024, 512, bias=True, priors=self.priors)
        self.act7 = self.act()
        self.fc3 = BBBLinear(512, 128, bias=True, priors=self.priors)
        self.act8 = self.act()
        self.fc4 = BBBLinear(128, 64, bias=True, priors=self.priors)
        self.act8 = self.act()
        
        self.fc5 = BBBLinear(64, outputs, bias=True, priors=self.priors)
        

class BBBmymodel1Layer(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBmymodel1Layer, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        # self.conv1 = BBBConv2d(inputs, 24, 3, bias=True, priors=self.priors)
        # self.act1 = self.act()
        # self.pool1 = nn.MaxPool2d(6,6)
        # self.conv2 = BBBConv2d(24, 48, 3, bias=True, priors=self.priors)
        # self.act2 = self.act()
        # self.pool2 = nn.MaxPool2d(6,6)
    
    
        # self.flatten = nn.Flatten(1)
        
        # self.fc1 = BBBLinear(1920, 1024, bias=True, priors=self.priors)
        # self.act3 = self.act()
        
        # self.fc2 = BBBLinear(1024, outputs, bias=True, priors=self.priors)
        self.conv1 = BBBConv2d(inputs, 4, 4, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(4,4)
        
    
    
        self.flatten = nn.Flatten(1)
        
        self.fc1 = BBBLinear(100, outputs, bias=True, priors=self.priors)

        