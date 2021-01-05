import torch
import torch.nn as nn
import math
from random import random as rd

__all__ = [ 'DCNet2', 'dcnet2']

class DCNet2(nn.Module):

    def __init__(self, features, num_classes):
        super(DCNet2, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(64*10, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True)
        )
        self.top_layer = nn.Linear(2048, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv1d):
                #print(y)
                n = m.kernel_size[0] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(input_dim, batch_norm):
    layers = []
    in_channels = input_dim
    # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
    cfg = [64, 64, 'M', 64, 64, 'M']


    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=5)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def dcnet2(sobel=False, bn=True, out=1000):
    dim = 1
    model = DCNet2(make_layers(dim, bn), out)
    # print("a\n\n")
    # exit()
    return model