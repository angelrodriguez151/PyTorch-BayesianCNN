import torch.nn as nn
from layers.misc import FlattenLayer


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class nodropout1d(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(nodropout1d, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(inputs, 3, 3),
            nn.Softplus(),
            
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(894, 100),
            nn.Softplus(),
            nn.Linear(100, outputs),
    
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
