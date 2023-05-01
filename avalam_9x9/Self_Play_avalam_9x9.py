import numpy as np
print(np.__version__)


import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


import random
import math



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rows = 9
columns = 9
max_height = 5
action_size = 9*9*8
max_score = 10  
n_states_encoded = max_height * 2 + 1 

def create_action_dictionary():
    action_dict = {}
    index = 0
    for row in range(rows):
        for col in range(columns):
            for drow in range(-1, 2):
                for dcol in range(-1, 2):
                    if drow == 0 and dcol == 0:
                        continue
                    new_row = row + drow
                    new_col = col + dcol
                    if 0 <= new_row < rows and 0 <= new_col < columns:
                        action_dict[(row, col, new_row, new_col)] = index
                        index += 1
    return action_dict

action_dict = create_action_dictionary()
index_to_action = {index: action for action, index in action_dict.items()}




class ResNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden, device, board_size, actions_size, dropout_rate=0.5):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(n_states_encoded, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden, dropout_rate) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size, actions_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * board_size, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x += residual
        x = F.relu(x)
        return x
