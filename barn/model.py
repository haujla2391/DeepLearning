import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 16)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, out_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class CNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool3 = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy = torch.zeros(1, *in_dim) 
            dummy = self.pool1(F.relu(self.conv1(dummy)))
            dummy = self.pool2(F.relu(self.conv2(dummy)))
            dummy = self.pool3(F.relu(self.conv3(dummy)))
            self.flatten_dim = dummy.numel()

        self.fc = FC(
            in_dim=self.flatten_dim,
            out_dim=out_dim,
        )


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CNN_small(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(CNN_small, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        # self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        # self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')



    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        x = self.fc2(x)

        return x