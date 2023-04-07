import sys
import torch
import copy
import h5py
import matplotlib.pyplot as plt
import seaborn
import wandb
from lifelines import *
import pandas as pd
import argparse

from modules.train import train
from modules.Encoder import UniTrans
from modules.utils import setup_seed
from modules.options import parse_args
from modules.EncoderLayer import EncoderLayer
from modules.MultiHeadedAttention import MultiHeadedAttention
from modules.PositionwiseFeedForward import PositionwiseFeedForward

seaborn.set_context(context="talk")
sys.path.append("..")


def main(features, image_ids, labels, num_features):
    c = copy.deepcopy
    attn = MultiHeadedAttention(opt.num_heads, opt.d_model, opt.drop_prob)
    ff = PositionwiseFeedForward(opt.d_model, opt.d_ff, opt.drop_prob)
    encoder_layer = EncoderLayer(opt.d_model, c(attn), c(ff), opt.drop_prob)
    encoder = UniTrans(opt.max_time, opt.in_channel, opt.out_dim, encoder_layer, opt.N, opt.d_model, opt.drop_prob, num_features).cuda()
    if opt.data_parallel:
        encoder = torch.nn.DataParallel(encoder).cuda()
    train(opt, features, image_ids, labels, encoder)


if __name__ == '__main__':
    opt = parse_args()
    setup_seed(seed=42)

    wandb.init(config=opt, project='MV-MS', entity='xinz', name='newnewcs_fold'+opt.data_fold)

    train_data = pd.read_csv(opt.data_dir + opt.data_fold + '/train_features_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
    val_data = pd.read_csv(opt.data_dir + opt.data_fold + '/val_features_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
    test_data = pd.read_csv(opt.data_dir + opt.data_fold + '/test_features_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
    train_features, val_features, test_features = train_data[:, 0:-1], val_data[:, 0:-1], test_data[:, 0:-1]
    train_img, val_img, test_img = train_data[:, -1], val_data[:, -1], test_data[:, -1]
    features = [train_features, val_features, test_features]
    image_ids = [train_img, val_img, test_img]

    train_labels = pd.read_csv(opt.data_dir + opt.data_fold + '/train_labels_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
    val_labels = pd.read_csv(opt.data_dir + opt.data_fold + '/val_labels_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
    test_labels = pd.read_csv(opt.data_dir + opt.data_fold + '/test_labels_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
    labels = [train_labels, val_labels, test_labels]

    num_features = train_features.shape[1]
    print('train features shape', train_features.shape)
    print('train labels shape', train_labels.shape)
    print('val features shape', val_features.shape)
    print('val labels shape', val_labels.shape)
    print('test features shape', test_features.shape)
    print('test labels shape', test_labels.shape)
    print()

    print('num features', num_features)
    total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
    print('total data', total_data)
    print()

    print('train max label', train_labels[:, 0].max())
    print('train min label', train_labels[:, 0].min())
    print('val max label', val_labels[:, 0].max())
    print('val min label', val_labels[:, 0].min())
    print('test max label', test_labels[:, 0].max())
    print('test min label', test_labels[:, 0].min())

    print('coeffs of losses', opt.coeff0, opt.coeff1, opt.coeff2, opt.coeff4, opt.coeff3)

    main(features, image_ids, labels, num_features)

    print('coeffs of losses', opt.coeff0, opt.coeff1, opt.coeff2, opt.coeff4, opt.coeff3)


