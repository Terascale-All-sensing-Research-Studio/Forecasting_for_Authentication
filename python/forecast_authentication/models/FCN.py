import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, data_len, num_features=4, num_class=2):
        super(FCN, self).__init__()
        self.num_class = num_class

        self.c1 = nn.Conv1d(num_features, 128, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(128)

        self.c2 = nn.Conv1d(128, 256, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(256)

        self.c3 = nn.Conv1d(256, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc = nn.Linear(data_len-13, num_class)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.c1(x)        
        x = self.relu(self.bn1(x))

        x = self.c2(x)
        x = self.relu(self.bn2(x))

        x = self.c3(x)
        x = self.relu(self.bn3(x))
        x = x.transpose(1, 2)

        x = torch.mean(x, 2)
        x = self.fc(x.reshape(x.size()[0], -1)) 

        return x