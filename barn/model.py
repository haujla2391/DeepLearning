import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):

    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        layers = []
        prev = in_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev, layer_size))
            prev = layer_size

        layers.append(nn.Linear(prev, out_dim))

        self.layers = nn.ModuleList(layers)
        self.num_hidden_layers = num_hidden_layers

    def forward(self, x):

        x = x.view(x.size(0), -1)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layers[self.num_hidden_layers](x)

class CNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(CNN, ).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten_dim = 64 * 63 * 47

        self.fc = FC(
            in_dim=self.flatten_dim,
            out_dim=out_dim,
            num_hidden_layers=256,
            layer_size=256
        )


    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pool2(x)

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