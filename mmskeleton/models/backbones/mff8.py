import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph


class MFFNet8(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}


        self.tcn_in = tcn(in_channels, 32, 3, 1, 1, residual=False)

        self.tcn_0 = nn.ModuleList((
            tcn(32, 32, 3, 1, 1),
            tcn(32, 32, 3, 1, 2),
            tcn(32, 32, 3, 1, 4),
        ))

        self.A_0 =  nn.Parameter(A + 0.01*torch.ones(self.A.size()))

        self.gcn_0 = nn.ModuleList((
            gcn(64, 32, 3),
            gcn(64, 32, 3),
            gcn(64, 64, 3),
        ))

        self.conv_fusion_0 = nn.ModuleList((
            nn.Conv2d(64, 64, 1),
            nn.Conv2d(64, 64, 1),
            nn.Conv2d(64, 64, 1),
        ))

        self.tcn_1 = nn.ModuleList((
            tcn(64, 64, 3, 1, 1),
            tcn(64, 64, 3, 1, 2),
            tcn(64, 64, 3, 1, 4),
        ))

        self.A_1 =  nn.Parameter(A + 0.01*torch.ones(self.A.size()))

        self.gcn_1 = nn.ModuleList((
            gcn(128, 64, 3),
            gcn(128, 64, 3),
            gcn(128, 128, 3),
        ))

        self.conv_fusion_1 = nn.ModuleList((
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(128, 128, 1),
        ))

        self.tcn_2 = nn.ModuleList((
            tcn(128, 128, 3, 1, 1),
            tcn(128, 128, 3, 1, 2),
            tcn(128, 128, 3, 1, 4),
        ))

        self.A_2 =  nn.Parameter(A + 0.01*torch.ones(self.A.size()))

        self.gcn_2 = nn.ModuleList((
            gcn(256, 128, 3),
            gcn(256, 128, 3),
            gcn(256, 256, 3),
        ))

        self.conv_fusion_2 = nn.ModuleList((
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(256, 256, 1),
        ))

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        x = self.tcn_in(x)

        feats = []
        for tcn in self.tcn_0:
            x = tcn(x)
            feats.append(x)

        feats = feats[::-1]
        for gcn, conv_f, f_up, in zip(self.gcn_0, self.conv_fusion_0, feats):
            x = conv_f(torch.cat([x, f_up], 1))
            x = gcn(x, self.A_0)

        feats = []
        for tcn in self.tcn_1:
            x = tcn(x)
            feats.append(x)

        feats = feats[::-1]
        for gcn, conv_f, f_up, in zip(self.gcn_1, self.conv_fusion_1, feats):
            x = conv_f(torch.cat([x, f_up], 1))
            x = gcn(x, self.A_1)

        feats = []
        for tcn in self.tcn_2:
            x = tcn(x)
            feats.append(x)

        feats = feats[::-1]
        for gcn, conv_f, f_up, in zip(self.gcn_2, self.conv_fusion_2, feats):
            x = conv_f(torch.cat([x, f_up], 1))
            x = gcn(x, self.A_2)

            
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0, residual=True):
        super().__init__()

        padding = ((kernel_size - 1)*dilation // 2, 0)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size, 1),
                (stride, 1),
                padding,
                (dilation, 1)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.tcn(x)
        x = x + res

        return self.relu(x)

class gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super().__init__()

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, _ = self.gcn(x, A)
        x = x + res

        return self.relu(x)


class st_gcn_block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A