import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph


class MFFNet5(nn.Module):
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

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # build networks
        # spatial_kernel_size = A.size(0)
        # temporal_kernel_size = 9
        # kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn_block(in_channels,
        #                  64,
        #                  kernel_size,
        #                  1,
        #                  residual=False,
        #                  **kwargs0),
        #     st_gcn_block(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn_block(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn_block(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn_block(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn_block(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn_block(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn_block(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        #     st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        # ))

        self.tcn0 = tcn(in_channels, 64, 1, 1)
        self.tcn1 = tcn(64, 64, 3, 1)
        self.tcn2 = tcn(64, 64, 3, 2)
        self.tcn3 = tcn(64, 64, 3, 3)
        self.tcn4 = tcn(64, 64, 3, 4)

        self.gcn_s1 = gcn(64, 64, 3)
        self.gcn_s2 = gcn(64, 64, 3)
        self.gcn_s3 = gcn(64, 64, 3)

        self.gcn_m1 = gcn(64, 64, 3)
        self.gcn_m2 = gcn(64, 64, 3)
        self.gcn_m3 = gcn(64, 64, 3)

        self.gcn_l1 = gcn(64, 64, 3)
        self.gcn_l2 = gcn(64, 64, 3)
        self.gcn_l3 = gcn(64, 64, 3)

        self.gcn_fusion = nn.ModuleList((
            tcn(64*9, 256, 1, 1),
            gcn(256, 128, 3),
            gcn(128, 128, 3),
        ))

        self.As = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(3)
            ])

        self.Aw = nn.ParameterList([
                nn.Parameter(0.01*torch.ones(self.A.size()))
                for i in range(3)
            ])

        self.Am = nn.ParameterList([
                nn.Parameter(0.1*torch.ones(self.A.size()))
                for i in range(3)
            ])

        self.Al = nn.ParameterList([
                nn.Parameter(0.1*torch.ones(self.A.size()))
                for i in range(3)
            ])

        self.Af = nn.ParameterList([
                nn.Parameter(0.1*torch.ones(self.A.size()))
                for i in range(2)
            ])


        # fcn for prediction
        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        x = self.tcn0(x)
        x = self.tcn1(x)

        x = self.tcn2(x)
        feat_s1 = self.gcn_s1(x, self.As[0]*self.A + self.Aw[0])
        feat_s2 = self.gcn_s2(x, self.As[1]*self.A + self.Aw[1])
        feat_s3 = self.gcn_s3(x, self.As[2]*self.A + self.Aw[2])

        x = self.tcn3(x)
        feat_m1 = self.gcn_m1(x, self.Am[0])
        feat_m2 = self.gcn_m2(x, self.Am[1])
        feat_m3 = self.gcn_m3(x, self.Am[2])
        
        x = self.tcn4(x)
        feat_l1 = self.gcn_l1(x, self.Al[0])
        feat_l2 = self.gcn_l2(x, self.Al[1])
        feat_l3 = self.gcn_l3(x, self.Al[2])

        x = torch.cat([feat_s1, feat_s2, feat_s3, feat_m1, feat_m2, feat_m3, feat_l1, feat_l2, feat_l3], 1)

        x = self.gcn_fusion[0](x)
        x = self.gcn_fusion[1](x, self.Af[0])
        x = self.gcn_fusion[2](x, self.Af[1])

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
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, dropout=0):

        super().__init__()

        padding = ((dilation*(kernel_size-1))//2, 0)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size, 1),
                (stride, 1),
                padding,
                dilation
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        res = self.residual(x)
        x = self.tcn(x)
        x = x + res

        return x

class gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):

        super().__init__()

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size)

        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x, A):
        res = self.residual(x)
        x, _ = self.gcn(x, A)
        x = x + res

        return x

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