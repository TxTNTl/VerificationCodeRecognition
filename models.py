import torch.nn as nn
import config


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # size = 160 * 80
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # size  = 80 * 40
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # size = 40 * 20
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # size = 20 * 10
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # size = 10 * 5
        self.layer5 = nn.Sequential(
            nn.Linear(512 * 10 * 5, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 5 * len(config.MAIN_SET['characters'])),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [batch_size, 512, 10, 5]
        x = x.view(x.size(0), -1)  # [batch_size, 512 * 10 * 5]
        x = self.layer5(x)
        return x
