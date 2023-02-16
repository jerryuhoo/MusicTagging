import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import sys
sys.path.append(".")
from modules import Conv_1d, Conv_2d
# class CNN(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(CNN, self).__init__()

#         self.conv1 = nn.Conv1d(
#             in_channels=input_dim, out_channels=64, kernel_size=3, padding=1
#         )
#         self.bn1 = nn.BatchNorm1d(64)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

#         self.conv2 = nn.Conv1d(
#             in_channels=64, out_channels=128, kernel_size=3, padding=1
#         )
#         self.bn2 = nn.BatchNorm1d(128)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

#         self.fc1 = nn.Linear(128 * (313 // 4), 128)
#         self.output_layer = nn.Sequential(nn.Linear(128, num_classes), nn.Sigmoid())

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(self.bn1(x))
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu(self.bn2(x))
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.output_layer(x)

#         return x


class CRNN(nn.Module):
    '''
    Choi et al. 2017
    Convolution recurrent neural networks for music classification.
    Feature extraction with CNN + temporal summary with RNN
    '''
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=20,
                f_max=8000,
                n_mels=128,
                n_class=50):
        super(CRNN, self).__init__()

        
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, 64, pooling=(2,2))
        self.layer2 = Conv_2d(64, 128, pooling=(3,3))
        self.layer3 = Conv_2d(128, 128, pooling=(4,4))
        self.layer4 = Conv_2d(128, 128, pooling=(4,4))

        # RNN
        self.layer5 = nn.GRU(128, 32, 2, batch_first=True)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(32, n_class)

    def forward(self, x):
        x = x.unsqueeze(1) 
        # print(x.shape)

        # CCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # RNN
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.layer5(x)
        x = x[:, -1, :]

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x