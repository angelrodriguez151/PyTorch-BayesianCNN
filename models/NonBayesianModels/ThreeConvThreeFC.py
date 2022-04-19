import torch.nn as nn
from layers.misc import FlattenLayer


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class ThreeConvThreeFC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(ThreeConvThreeFC, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(inputs, 24, 3),
            nn.Softplus(),
            nn.MaxPool2d(6,6),
            nn.Dropout(),
            
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(2400,1024),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(1024, outputs),
    
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
