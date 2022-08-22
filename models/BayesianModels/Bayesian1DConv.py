import math
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv1d
from layers import BBB_LRT_Linear, BBB_LRT_Conv1d
from layers import FlattenLayer, ModuleWrapper

class BBBConv1(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBConv1, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv1d = BBB_LRT_Conv1d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv1d = BBB_Conv1d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")
        
        self.conv1 = BBBConv1d(inputs, 4, 8, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool1d(8)
        self.flatten = nn.Flatten(1)
        self.fc1 = BBBLinear(68, 32, bias=True, priors=self.priors)
        self.act4 = self.act()
        self.fc2 = BBBLinear(32, outputs, bias=True, priors=self.priors)


class BBBLinear(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBLinear, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.fc1 = BBBLinear(44, 64, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.fc2 = BBBLinear(64, outputs, bias=True, priors=self.priors)
        