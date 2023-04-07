import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Survival analysis')
    parser.add_argument('--max_time', type=int, default=100, help='max number of months')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--in_channel', type=int, default=3, help='number of input channels of CNN')
    parser.add_argument('--out_dim', type=int, default=10, help='number of output latent feature from CNN')
    parser.add_argument('--N', type=int, default=4, help='number of modules')
    parser.add_argument('--num_heads', type=int, default=4)  # number of heads, to MultiHeadedAttention   {1,2,4,8}
    parser.add_argument('--d_model', type=int, default=512)  # d_model, to MultiHeadedAttention / PositionwiseFeedForward / EncoderLayer / Encoder  {256,512}
    parser.add_argument('--d_ff', type=int, default=2048)  # hidden layer of PositionwiseFeedForward
    parser.add_argument('--train_batch_size', type=int, default=16)  # {4,8,16,32}
    parser.add_argument('--drop_prob', type=float, default=0.1)  # {0.0, 0.1, 0.3, 0.5}
    parser.add_argument('--lr', type=float, default=1e-4)  # {1e-4, 5e-4, 1e-3}
    parser.add_argument('--lr_decay_epoch', type=int, default=300)
    parser.add_argument('--lr_decay_factor', type=float, default=1e-3)
    parser.add_argument('--delta_t', type=float, default=10)  #margin survival time for censoring
    parser.add_argument('--coeff0', type=float, default=1)  #mean loss
    parser.add_argument('--coeff1', type=float, default=1)  # var loss
    parser.add_argument('--coeff2', type=float, default=1)  # Disc loss
    parser.add_argument('--coeff3', type=float, default=1)  # mean2 loss
    parser.add_argument('--coeff4', type=float, default=100)  # CE loss
    parser.add_argument('--data_dir', type=str, default='data/cro_val_')
    parser.add_argument('--image_dir', type=str, default='data/img_linear_separate/')
    parser.add_argument('--data_fold', type=str, default='0')
    parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints/cro_val_')
    parser.add_argument('--report_interval', type=int, default=1)
    parser.add_argument('--data_parallel', action='store_true', help='use data parallel?')
    parser.add_argument('--pred_method', type=str, choices=['mean', 'median'], default='mean')
    opt = parser.parse_args()
    print(opt)
    return opt
