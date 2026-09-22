import torch.nn as nn
import torch.nn.functional as F

class AirfoilNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 3)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x