import torch.nn as nn
from collections import OrderedDict

class LastLayer(nn.Module):
    
    def __init__(self):
        super(LastLayer, self).__init__()
        self.out_features = 2
        self.fc = nn.Sequential(OrderedDict([
            ('fc_Linear1', nn.Linear(512, 8)),
            ('activation', nn.ReLU()),
            ('fc_Linear2', nn.Linear(8, self.out_features))
        ]))
        
    def forward(self, x):
        return self.fc(x)