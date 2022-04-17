import math
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper

class BBB3Conv3FC(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBB3Conv3FC, self).__init__()

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
        self.pool1 = nn.MaxPool2d(6,6)
    
        self.flatten = nn.Flatten(1)
        
        self.fc1 = BBBLinear(10 * 10 * 8, 128, bias=True, priors=self.priors)
        self.act3 = self.act()
        self.fc2 = BBBLinear(128, outputs, bias=True, priors=self.priors)