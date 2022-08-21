import torch.nn as nn
from layers.misc import FlattenLayer


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class dropout(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(dropout, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(inputs, 8, 3),
            nn.Softplus(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            
            nn.Conv2d(8, 16, 3),
            nn.Softplus(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            
            nn.Conv2d(16, 32, 3),
            nn.Softplus(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1152, 512),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(512, 32),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(32, outputs),
    
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class dropout1(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(dropout1, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(inputs, 24, 3),
            nn.Softplus(),
            nn.MaxPool2d(6,6),
            
            nn.Conv2d(24, 48, 3),
            nn.Softplus(),
            nn.MaxPool2d(6,6),

            
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1920, 512),
            nn.Softplus(),
            nn.Linear(512, 32),
            nn.Softplus(),
            nn.Linear(32, outputs),
    
        )
      
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
