import torch.nn as nn
from layers.misc import FlattenLayer


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class dropout1d(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(dropout1d, self).__init__()
        self.features = nn.Sequential(
            
            nn.Conv1d(inputs,4, 8),
            nn.Softplus(),
            nn.Dropout(),
            nn.MaxPool1d(2),
            nn.Conv1d(4,8, 8),
            nn.Softplus(),
            nn.Dropout(),
            nn.MaxPool1d(2),
            nn.Conv1d(8,16, 8),
            nn.Softplus(),
            nn.Dropout(),
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(5792,1000),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(1000,100),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(100, outputs),
    
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
