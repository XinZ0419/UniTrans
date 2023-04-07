import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules.utils import clones, LayerNorm
import math


# UniTrans ============================================================================================================
class UniTrans(nn.Module):

    def __init__(self, max_time, in_channel, out_dim, layer, N, d_model, dropout, num_features):
        super(ConvTrans, self).__init__()
        self.max_time = max_time
        self.mini_cnn = MinimalistCNN(in_channel, out_dim)
        self.pre_cnn = pretrainedCNN()
        self.fc = nn.Sequential(nn.Linear(num_features, 32), nn.ReLU(), nn.Linear(32, out_dim), nn.ReLU())
        self.trans_encoder = Encoder(layer, N, d_model, dropout, 2*out_dim)

    def forward(self, feature, image):
        # latent_feature = self.mini_cnn(image)
        latent_feature_1 = self.pre_cnn(image)
        latent_feature_2 = self.fc(feature.to(torch.float32))
        x = torch.cat((latent_feature_1, latent_feature_2), -1)
        x = torch.stack(self.max_time * [x], dim=1)
        x = self.trans_encoder(x)
        return x


# CNN ==================================================================================================================
class MinimalistCNN(nn.Module):

    def __init__(self, in_channel=3, out_dim=32):
        super(MinimalistCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=5, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.fcout = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fcout(x)


# ----------------------------------------------------------------------------------------------------------------------
class pretrainedCNN(nn.Module):
    def __init__(self, opt=None):
        super(pretrainedCNN, self).__init__()
        self.img_org = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

    def forward(self, img):
        x = self.img_org(img)

        return x


# Transformer ==========================================================================================================
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N, d_model, dropout, num_features):
        super(Encoder, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.final_layer = TranFinalLayer(d_model)

    def forward(self, x, mask=None):
        """Pass the input (and mask) through each layer in turn."""
        x = x.to(torch.float32)
        x = self.position_encode(self.src_embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer(x)


# initial embedding for raw input
class SrcEmbed(nn.Module):
    def __init__(self, input_dim, d_model):
        super(SrcEmbed, self).__init__()
        self.w = nn.Linear(input_dim, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.w(x))


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TranFinalLayer(nn.Module):  # final layer for the transformer
    def __init__(self, d_model):
        super(TranFinalLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model // 2)
        self.norm = LayerNorm(d_model // 2)
        self.w_2 = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.norm(x)
        x = self.w_2(x)
        # return torch.sigmoid(x.squeeze(-1))
        # return F.softmax(x.squeeze(-1), dim=-1)
        return x
# ======================================================================================================================
