import torch
import torch.nn as nn
import torch.nn.functional as F

# Define EMNIST NN Model
class EMNIST_NN(nn.Module):
    def __init__(self):
        super(EMNIST_NN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # Change input channels to 1 for grayscale images
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=5)
        self.fc1 = nn.Linear(15 * 4 * 4, 128)  # Adjusted size for the flattened image after convolutions and pooling
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 47)  # EMNIST has 47 classes instead of 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 15 * 4 * 4)  # Flatten the tensor before passing to fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
