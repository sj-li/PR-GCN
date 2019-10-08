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
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.conv_shift_1 = nn.Conv2d(3, 64, 1)
        self.gcn_shift_1 = gcn_o(64, 128, 3)
        self.gcn_shift_2= gcn_o(128, 128, 3)
        self.conv_shift_2 = nn.Conv2d(128, 3, 1)

        self.A_shift_1 = nn.Parameter(A + 0.0001*torch.ones(A.size()))
        self.A_shift_2 = nn.Parameter(A + 0.0001*torch.ones(A.size()))

        self.tcn_motion_in = tcn(in_channels, 64, 3, 1, 1, residual=False)
        self.tcn_pos_in_1 = tcn(in_channels, 64, 3, 1, 1, residual=False)
        self.tcn_pos_in_2 = tcn(in_channels, 64, 3, 1, 1, residual=False)
        self.conv_fusion_in = nn.Conv2d(128, 64, 1)

        self.tcn = nn.ModuleList((
            tcn(64, 64, 3, 1, 1, residual=False),
            tcn(64, 64, 3, 2, 1, residual=False),
            tcn(64, 64, 3, 1, 1, residual=False),
            tcn(64, 64, 3, 3, 1, residual=False)
        ))

        self.gcn = nn.ModuleList((
            gcn_o(128, 64, 3),
            gcn_o(128, 64, 3),
            gcn_o(128, 64, 3),
            gcn_o(256, 256, 3)
        ))

        self.As =  nn.ParameterList([
                nn.Parameter(A + 0.0001*torch.ones(A.size()))
                for i in self.gcn
            ])

        self.conv_fusion = nn.ModuleList((
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(128, 256, 1)
        ))

        self.gcn_gather = gcn(256, 256)
        self.A_gather = nn.Parameter(A + 0.0001*torch.ones(A.size()))


        # self.conv_trans = nn.Conv2d(256, 128, 1)
        # self.conv_gather = nn.Conv2d(256, 256, 1)
        # self.soft = nn.Softmax(-2)

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

        shift = self.conv_shift_1(x)
        shift = self.gcn_shift_1(shift, self.A_shift_1)
        shift = self.gcn_shift_2(shift, self.A_shift_2)
        shift = self.conv_shift_2(shift)
        shift[:, 2, :, :] = 0
        
        x = x + shift

        motion = torch.zeros_like(x)
        motion[:,:,1:,:] = x[:,:,1:,:] - x[:,:,:-1,:]

        # forwad
        motion = self.tcn_motion_in(motion)
        x_ = self.tcn_pos_in_1(x)
        x = self.tcn_pos_in_2(x)
        x = self.conv_fusion_in(torch.cat([x, motion], 1))

        feats = []
        for tcn in self.tcn:
            x = tcn(x)
            feats.append(x)

        feats = feats[::-1]

        x = x_
        for gcn, conv_f, A_, f, in zip(self.gcn, self.conv_fusion, self.As, feats):
            if x.shape[2] > f.shape[2]:
                x = self.scale_T(x, f.shape[2])
            if x.shape[2] < f.shape[2]:
                f = self.scale_T(f, x.shape[2])
            x = conv_f(torch.cat([x, f], 1))
            x = gcn(x, A_)

        x = self.gcn_gather(x, self.A_gather)

        # x_ = self.conv_trans(x)
        # x1 = x_.permute(0, 3, 1, 2).contiguous().view(N*M, V, -1)
        # x2 = x_.view(N*M, -1, V)
        # A = self.soft(torch.matmul(x1, x2) / x1.size(-1))
        # x = x.view(N*M, -1, V)
        # x = self.conv_gather(torch.matmul(x, A).view(N*M, 256, -1, V))
            
        # global pooling
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def scale_T(self, x, newT):
        N, C, T, V = x.shape
        x = x.view(N, C, newT, T//newT, V)
        x, _ = torch.max(x, 3)
        x = x.squeeze()

        return x
        

class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0, residual=True):
        super().__init__()

        padding = ((kernel_size - 1)*dilation // 2, 0)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True),
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

class gcn_o(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super().__init__()

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)

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
        x = self.bn(x) + res

        return self.relu(x)

class gcn(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=False, num_subset=3):
        super().__init__()

        # self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)

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

        self.num_subset = num_subset

        # self.conv_trans_1 = nn.Conv2d(in_channels, 64, 1)
        # self.conv_trans_2 = nn.Conv2d(in_channels, 64, 1)
        # self.conv_gather = nn.Conv2d(in_channels, out_channels, 1)

        inter_channels = out_channels//2
        self.inter_c = inter_channels
         
        self.conv_a = nn.ModuleList()
        # self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            # self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        self.soft = nn.Softmax(-2)

    def forward(self, x, A):
        # res = self.residual(x)

        # N, C, T, V = x.size()

        # x1 = self.conv_trans_1(x).permute(0, 3, 1, 2).contiguous().view(N, V, -1)
        # x2 = self.conv_trans_2(x).view(N, -1, V)
        # PA = self.soft(torch.matmul(x1, x2) / x1.size(-1))
        # x = x.view(N, -1, V)
        # x = self.conv_gather(torch.matmul(x, PA).view(N, -1, T, V))
        # x = self.bn(x) + res

        # return self.relu(x)
        N, C, T, V = x.size()
        y = None
        for i in range(self.num_subset):
            m = self.conv_a[i](x)
            A1 = m.permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = m.view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        # y = self.bn(y)
        # y += self.residual(x)
        return y


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