import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph


class MFFNet7(nn.Module):
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

        self.gcn_in = gcn(in_channels, 64, 3, 1, residual=False)
        self.A_in = nn.Parameter(A + 0.1*torch.ones(self.A.size()))

        self.gcn_in_1 =  gcn(64, 64, 3, 1)
        self.gcn_in_2 =  gcn(64, 64, 3, 1)
        self.gcn_in_3 =  gcn(64, 64, 3, 1)
        self.gcn_in_4 =  gcn(64, 64, 3, 1)

        self.A_in_1 = nn.Parameter(A + 0.1*torch.ones(self.A.size()))
        self.A_in_2 = nn.Parameter(A + 0.1*torch.ones(self.A.size()))
        self.A_in_3 = nn.Parameter(A + 0.1*torch.ones(self.A.size()))
        self.A_in_4 = nn.Parameter(A + 0.1*torch.ones(self.A.size()))

        self.gcn_out_1 =  gcn(128, 256, 3, 1)
        self.gcn_out_2 =  gcn(128, 64, 3, 1)
        self.gcn_out_3 =  gcn(128, 64, 3, 1)
        self.gcn_out_4 =  gcn(128, 64, 3, 1)        

        self.A_out_1 = nn.Parameter(A + 0.1*torch.ones(self.A.size()))
        self.A_out_2 = nn.Parameter(A + 0.1*torch.ones(self.A.size()))
        self.A_out_3 = nn.Parameter(A + 0.1*torch.ones(self.A.size()))
        self.A_out_4 = nn.Parameter(A + 0.1*torch.ones(self.A.size()))

        self.tcn_1 = tcn(64, 128, 3, 1, 1)
        self.tcn_2 = tcn(128, 128, 3, 1, 2)
        self.tcn_3 = tcn(128, 128, 3, 1, 4)
        self.tcn_4 = tcn(128, 64, 3, 1, 8)

        self.conv_f_1 = nn.Conv2d(128, 128, 1)
        self.conv_f_2 = nn.Conv2d(128, 128, 1)
        self.conv_f_3 = nn.Conv2d(128, 128, 1)
        self.conv_f_4 = nn.Conv2d(128, 128, 1)


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

        x = self.gcn_in(x, self.A_in)

        x1 = self.gcn_in_1(x, self.A_in_1)
        x2 = self.gcn_in_2(x1, self.A_in_2)
        x3 = self.gcn_in_3(x2, self.A_in_3)
        x4  = self.gcn_in_4(x3, self.A_in_4)

        x = x4
        x = self.tcn_1(x)
        x = self.tcn_2(x)
        x = self.tcn_3(x)
        x = self.tcn_4(x)

        x3 = self.gcn_out_4(self.conv_f_4(torch.cat([x, x4], 1)), self.A_out_4)
        x2 = self.gcn_out_3(self.conv_f_3(torch.cat([x, x3], 1)), self.A_out_3)
        x1 = self.gcn_out_2(self.conv_f_2(torch.cat([x, x2], 1)), self.A_out_2)
        x  = self.gcn_out_1(self.conv_f_1(torch.cat([x, x1], 1)), self.A_out_1)
            
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


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