import torch
from torch import nn

from .models import register_backbone
from .blocks import (MaskedConv1D, LayerNorm, TemporalMaxer, TemporalMiner, TemporalMaxMiner)


@register_backbone("convPooler")
class ConvPoolerBackbone(nn.Module):
    """
        A backbone that combines convolutions with TemporalMaxer
    """

    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch=(2, 5),           # (#convs, #branch TemporalMaxer)
        scale_factor=2,       # dowsampling rate for the branch
        with_ln=False,        # if to attach layernorm after conv
        **kwargs,
    ):
        super().__init__()
        assert len(arch) == 2
        self.n_in = n_in
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(
                n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in # # FIXME: was n_embd, not //2
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks, # FIXME: was n_embd, not //2
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)))

            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd)) # FIXME: was n_embd, not //2
            else:
                self.embd_norm.append(nn.Identity())

        # main branch using TemporalMaxer
        self.branch_maxpool = nn.ModuleList()
        for idx in range(arch[1]):
            self.branch_maxpool.append(TemporalMaxer(kernel_size=3,
                                             stride=scale_factor,
                                             padding=1,
                                             n_embd=n_embd))
        
        self.branch_minpool = nn.ModuleList()
        for idx in range(arch[1]):
            self.branch_minpool.append(TemporalMiner(kernel_size=3,
                                             stride=scale_factor,
                                             padding=1,
                                             n_embd=n_embd))
            
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                 ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # print("shape of x: ", x.shape)

        # prep for outputs
        out_feats = (x, )
        # out_feats = (torch.cat([x, x], dim=1), )
        out_masks = (mask, )

        # print("shape of feats: ", out_feats[0].shape)

        x_max, mask_max = x, mask
        x_min, mask_min = x, mask

        # main branch with downsampling
        for idx in range(len(self.branch_maxpool)):
            x, mask = self.branch_maxpool[idx](x, mask)

            # x_max, mask_max = self.branch_maxpool[idx](x_max, mask_max)
            # x_min, mask_min = self.branch_minpool[idx](x_min, mask_min)

            # x = torch.cat([x_max, x_min], dim=1)
            # mask = mask_max
            
            # print("shape of x: ", x.shape)

            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks
