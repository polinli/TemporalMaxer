import torch
import torch.nn as nn
import random

class TemporalShift_random(nn.Module):
    def __init__(self, n_div=64, shift_dist=1, inplace=False, channel_size=2304):
        super(TemporalShift_random, self).__init__()
        # self.net = net
        self.fold_div = n_div
        self.shift_dist = shift_dist
        self.inplace = inplace
        self.channels_range = list(range(channel_size))  # feature_channels
        if inplace:
            print('=> Using in-place shift...')
        # print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        # self.fold_div = n_div
        x = self.shift(x, fold_div=self.fold_div, shift_dist=self.shift_dist , inplace=self.inplace, channels_range =self.channels_range)
        return x

    @staticmethod
    def shift(x, fold_div, shift_dist, inplace, channels_range):
        x = x.permute(0, 2, 1)   # [B,C,T] --> [B, T, C]
        # set_trace()
        n_batch, T, c = x.size()
        # nt, c, h, w = x.size()
        # n_batch = nt // n_segment
        # x = x.view(n_batch, n_segment, c, h, w)
        # x = x.view(n_batch, T, c, h, w)
        fold = c // fold_div
        all = random.sample(channels_range, fold*2)
        forward = sorted(all[:fold])
        backward = sorted(all[fold:])
        fixed = list(set(channels_range) - set(all))
        # fold = c // fold_div

        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            # out = torch.zeros_like(x)
            # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            # out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
            out = torch.zeros_like(x)
            out[:, :-shift_dist, forward] = x[:, shift_dist:, forward]  # shift left
            out[:, shift_dist:, backward] = x[:, :-shift_dist, backward]  # shift right
            out[:, :, fixed] = x[:, :, fixed]  # not shift

        # return out.view(nt, c, h, w)
        return out.permute(0, 2, 1)