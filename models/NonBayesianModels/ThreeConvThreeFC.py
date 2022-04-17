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
            nn.Conv2d(inputs, 32, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(128, 128, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(128, 256, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 256, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1,3),
            nn.Linear(2 * 2 * 256, 1024),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Dropout(),
            nn.Linear(128, outputs),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = self.classifier(x)
        return x
