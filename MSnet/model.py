import torch
import torch.nn as nn
import torch.nn.functional as F
class MSnet_melody(nn.Module):
    def __init__(self):
        super(MSnet_melody, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 5, padding=2),
            nn.SELU()
            )
        self.pool1 = nn.MaxPool2d((4,1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.SELU()
            )
        self.pool2 = nn.MaxPool2d((4,1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.SELU()
            )
        self.pool3 = nn.MaxPool2d((5,1), return_indices=True)

        self.bottom = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, 5, padding=(0,2)),
            nn.SELU()
            )

        self.up_pool3 = nn.MaxUnpool2d((5,1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.SELU()
            )

        self.up_pool2 = nn.MaxUnpool2d((4,1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.SELU()
            )

        self.up_pool1 = nn.MaxUnpool2d((4,1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 5, padding=2),
            nn.SELU()
            )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        c1, ind1 = self.pool1(self.conv1(x))
        c2, ind2 = self.pool2(self.conv2(c1))
        c3, ind3 = self.pool3(self.conv3(c2))
        bm = self.bottom(c3)
        u3 = self.up_conv3(self.up_pool3(c3, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        output = self.softmax(torch.cat((bm, u1), dim=2))

        return output, bm
class MSnet_vocal(nn.Module):
    def __init__(self):
        super(MSnet_vocal, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 5, padding=2),
            nn.SELU()
            )
        self.pool1 = nn.MaxPool2d((4,1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.SELU()
            )
        self.pool2 = nn.MaxPool2d((4,1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.SELU()
            )
        self.pool3 = nn.MaxPool2d((4,1), return_indices=True)

        self.bottom = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, 5, padding=(0,2)),
            nn.SELU()
            )

        self.up_pool3 = nn.MaxUnpool2d((4,1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.SELU()
            )

        self.up_pool2 = nn.MaxUnpool2d((4,1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.SELU()
            )

        self.up_pool1 = nn.MaxUnpool2d((4,1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 5, padding=2),
            nn.SELU()
            )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        c1, ind1 = self.pool1(self.conv1(x))
        c2, ind2 = self.pool2(self.conv2(c1))
        c3, ind3 = self.pool3(self.conv3(c2))
        bm = self.bottom(c3)
        u3 = self.up_conv3(self.up_pool3(c3, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        output = self.softmax(torch.cat((bm, u1), dim=2))

        return output, bm