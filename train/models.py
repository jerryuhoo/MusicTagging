"""
code from https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import sys
import numpy as np

sys.path.append(".")
from modules import (
    Conv_1d,
    ResSE_1d,
    Conv_2d,
    Res_2d,
    Conv_V,
    Conv_H,
    HarmonicSTFT,
    Res_2d_mp,
)

# from attention_modules import (
#     BertConfig,
#     BertEncoder,
#     BertEmbeddings,
#     BertPooler,
#     PositionalEncoding,
# )

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
    """
    Choi et al. 2017
    Convolution recurrent neural networks for music classification.
    Feature extraction with CNN + temporal summary with RNN
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        f_min=20,
        f_max=8000,
        n_mels=128,
        n_class=50,
        feature_type="log_mel",
        dropout=0.5,
        feature_extraction="log_mel",
    ):
        super(CRNN, self).__init__()
        if feature_type == "wav":
            self.spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
            )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, 64, pooling=(2, 2))
        self.layer2 = Conv_2d(64, 128, pooling=(3, 3))
        self.layer3 = Conv_2d(128, 128, pooling=(4, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(4, 4))

        # RNN
        self.layer5 = nn.GRU(128, 32, 2, batch_first=True)

        # Dense
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(32, n_class)

        self.feature_type = feature_type
        self.feature_extraction = feature_extraction

    def cqt_feature(self, x):
        cqt_list = []
        for i in range(x.shape[0]):
            cqt = librosa.cqt(
                np.asarray(x[i].cpu().squeeze()),
                sr=self.sample_rate,
                hop_length=self.hop_length,
                bins_per_octave=self.bins_per_octave,
                n_bins=self.n_bins,
            )
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            cqt_db = torch.from_numpy(cqt_db).to(x.device)
            cqt_list.append(cqt_db.unsqueeze(0))
        return torch.stack(cqt_list)

    def forward(self, x):
        if self.feature_type == "wav":
            if self.feature_extraction == "log_mel":
                x = self.spec(x)
                x = self.to_db(x)
                x = x.unsqueeze(1)
            elif self.feature_extraction == "cqt":
                x = self.cqt_feature(x)
            elif self.feature_extraction == "concat":
                x_mel = self.spec(x)
                x_mel = self.to_db(x_mel)
                x_mel = x_mel.unsqueeze(1)
                x_cqt = self.cqt_feature(x)
                x = torch.cat((x_mel, x_cqt), dim=2)
        else:
            x = x.unsqueeze(1)

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


class HarmonicCNN(nn.Module):
    """
    Won et al. 2020
    Data-driven harmonic filters for audio representation learning.
    Trainable harmonic band-pass filters, short-chunk CNN.
    """

    def __init__(
        self,
        n_channels=128,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=50,
        n_harmonic=6,
        semitone_scale=2,
        learn_bw="only_Q",
        dropout=0.5,
    ):
        super(HarmonicCNN, self).__init__()

        # Harmonic STFT
        self.hstft = HarmonicSTFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_harmonic=n_harmonic,
            semitone_scale=semitone_scale,
            learn_bw=learn_bw,
        )
        self.hstft_bn = nn.BatchNorm2d(n_harmonic)

        # CNN
        self.layer1 = Conv_2d(n_harmonic, n_channels, pooling=2)
        self.layer2 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer3 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer4 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer5 = Conv_2d(n_channels, n_channels * 2, pooling=2)
        self.layer6 = Res_2d_mp(n_channels * 2, n_channels * 2, pooling=(2, 3))
        self.layer7 = Res_2d_mp(n_channels * 2, n_channels * 2, pooling=(2, 3))

        # Dense
        self.dense1 = nn.Linear(n_channels * 2, n_channels * 2)
        self.bn = nn.BatchNorm1d(n_channels * 2)
        self.dense2 = nn.Linear(n_channels * 2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.hstft_bn(self.hstft(x))

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


class FCN(nn.Module):
    """
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=96,
        n_class=50,
        feature_type="log_mel",
        hop_length=256,
        bins_per_octave=12,
        n_bins=84,
        feature_extraction="log_mel",
        dropout=0.5,
    ):
        super(FCN, self).__init__()
        if feature_type == "wav":
            # Spectrogram
            self.spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
            )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.n_bins = n_bins
        self.feature_extraction = feature_extraction

        # # FCN short
        # self.layer1 = Conv_2d(1, 64, pooling=(2, 2))
        # self.layer2 = Conv_2d(64, 128, pooling=(2, 2))
        # self.layer3 = Conv_2d(128, 128, pooling=(2, 4))
        # self.layer4 = Conv_2d(128, 128, pooling=(4, 4))
        # self.layer5 = Conv_2d(128, 64, pooling=(4, 4))

        # FCN long
        self.layer1 = Conv_2d(1, 64, pooling=(2, 4))
        self.layer2 = Conv_2d(64, 128, pooling=(2, 4))
        self.layer3 = Conv_2d(128, 128, pooling=(2, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(3, 5))
        self.layer5 = Conv_2d(128, 64, pooling=(4, 4))

        # Dense
        if self.feature_extraction == "concat" and self.feature_type == "wav":
            self.dense = nn.Linear(64 * 2, n_class)
        else:
            self.dense = nn.Linear(64, n_class)
        self.dropout = nn.Dropout(dropout)

    def cqt_feature(self, x):
        cqt_list = []
        for i in range(x.shape[0]):
            cqt = librosa.cqt(
                np.asarray(x[i].cpu().squeeze()),
                sr=self.sample_rate,
                hop_length=self.hop_length,
                bins_per_octave=self.bins_per_octave,
                n_bins=self.n_bins,
            )
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            cqt_db = torch.from_numpy(cqt_db).to(x.device)
            cqt_list.append(cqt_db.unsqueeze(0))
        return torch.stack(cqt_list)

    def forward(self, x):
        if self.feature_type == "wav":
            if self.feature_extraction == "log_mel":
                x = self.spec(x)
                x = self.to_db(x)
                x = x.unsqueeze(1)
            elif self.feature_extraction == "cqt":
                x = self.cqt_feature(x)
            elif self.feature_extraction == "concat":
                x_mel = self.spec(x)
                x_mel = self.to_db(x_mel)
                x_mel = x_mel.unsqueeze(1)
                x_cqt = self.cqt_feature(x)
                x = torch.cat((x_mel, x_cqt), dim=2)
        else:
            x = x.unsqueeze(1)
        x = self.spec_bn(x)
        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x


class ShortChunkCNN(nn.Module):
    """
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    """

    def __init__(
        self,
        n_channels=96,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=96,
        n_class=50,
        feature_type="wav",
        hop_length=256,
        bins_per_octave=12,
        n_bins=84,
        feature_extraction="log_mel",
        dropout=0.5,
    ):
        super(ShortChunkCNN, self).__init__()

        if feature_type == "wav":
            # Spectrogram
            self.spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
            )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, n_channels, pooling=2)
        self.layer2 = Conv_2d(n_channels, n_channels, pooling=2)
        self.layer3 = Conv_2d(n_channels, n_channels * 2, pooling=2)
        self.layer4 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer5 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer6 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer7 = Conv_2d(n_channels * 2, n_channels * 4, pooling=2)

        # Dense
        self.dense1 = nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.feature_type = feature_type
        self.feature_extraction = feature_extraction

        self.n_bins = n_bins

    def cqt_feature(self, x):
        cqt_list = []
        for i in range(x.shape[0]):
            cqt = librosa.cqt(
                np.asarray(x[i].cpu().squeeze()),
                sr=self.sample_rate,
                hop_length=self.hop_length,
                bins_per_octave=self.bins_per_octave,
                n_bins=self.n_bins,
            )
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            cqt_db = torch.from_numpy(cqt_db).to(x.device)
            cqt_list.append(cqt_db.unsqueeze(0))
        return torch.stack(cqt_list)

    def forward(self, x):
        if self.feature_type == "wav":
            if self.feature_extraction == "log_mel":
                x = self.spec(x)
                x = self.to_db(x)
                x = x.unsqueeze(1)
            elif self.feature_extraction == "cqt":
                x = self.cqt_feature(x)
            elif self.feature_extraction == "concat":
                x_mel = self.spec(x)
                x_mel = self.to_db(x_mel)
                x_mel = x_mel.unsqueeze(1)
                x_cqt = self.cqt_feature(x)
                x = torch.cat((x_mel, x_cqt), dim=2)
        else:
            x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x
