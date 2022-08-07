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
            
            nn.Conv2d(inputs, 6, 4),
            nn.Softplus(),
            nn.MaxPool2d(2,2),
            nn.Dropout(),
            nn.Conv2d(6, 12, 4),
            nn.Softplus(),
            nn.MaxPool2d(2,2),
            nn.Dropout(),
            nn.Conv2d(12, 24, 4),
            nn.Softplus(),
            nn.MaxPool2d(2,2),
            nn.Dropout(),
            
            
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1536,512),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(512,128),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(128,16),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(16, outputs),
    
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
