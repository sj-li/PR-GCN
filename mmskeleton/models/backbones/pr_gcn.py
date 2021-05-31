import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(out_channels, 1, 1, 1, 0),
                                  nn.BatchNorm2d(1))
    def forward(self,x):
        x=self.conv(x)
        x=torch.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels*2, 1)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, 1)
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=torch.sigmoid(x)
        return x

# gcn in self.gcn, remove [::-1]
class PR_GCN(nn.Module):
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

        self.conv_shift_1 = nn.Conv2d(3, 32, 1)
        self.gcn_shift_1 = gcn(32, 32, 3, A)
        self.tcn_shift = tcn(32, 64, 3, 1, 1)
        self.gcn_shift_2= gcn(64, 64, 3, A)
        self.conv_shift_2 = nn.Conv2d(64, 3, 1)

        self.tcn_pos_in = tcn(in_channels, 64, 3, 1, 1)
        self.tcn_motion_in = tcn(in_channels, 64, 3, 1, 1)

        self.tcn = nn.ModuleList((
            tcn(64, 64, 3, 1, 1),
            tcn(64, 64, 3, 2, 1),
	    gcn(64, 64, 3, A),
            tcn(64, 64, 3, 1, 1),
            tcn(64, 64, 3, 3, 1)
        ))

        self.gcn = nn.ModuleList((
            gcn(128, 64, 3, A),
            gcn(128, 64, 3, A),
            gcn(128, 64, 3, A),
            gcn(128, 64, 3, A),
            gcn(128, 128, 3, A)
        ))


        # fcn for prediction
        self.gcn_end = gcn(128, 256, 3, A)
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        self.cSE_t = cSE(128)
        self.i = 0

    def PRM(self, x):
        shift = self.conv_shift_1(x)
        shift = self.gcn_shift_1(shift)
        shift = self.tcn_shift(shift)
        shift = self.gcn_shift_2(shift)
        shift = self.conv_shift_2(shift)
        shift[:, 2, :, :] = 0

        x = x + shift

        return x

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        x = self.PRM(x)

        motion = torch.zeros_like(x)
        motion[:,:,1:,:] = x[:,:,1:,:] - x[:,:,:-1,:]

        # forwad
        motion = self.tcn_motion_in(motion)
        x = self.tcn_pos_in(x)

        feats = []
        for tcn in self.tcn:
            motion = tcn(motion)
            feats.append(motion)

        for gcn, f, in zip(self.gcn, feats):
            if x.shape[2] > f.shape[2]:
                x = self.scale_T(x, f.shape[2])
            if x.shape[2] < f.shape[2]:
                f = self.scale_T(f, x.shape[2])
            x = gcn(torch.cat([x, f], 1))

        x = F.avg_pool2d(x, (5, 1))
        x = self.cSE_t(x)*x
        x = self.gcn_end(x)
        x = F.avg_pool2d(x, x.size()[2:])

        x, _ = x.view(N, M, -1, 1, 1).max(dim=1)

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, dropout=0, residual=False):
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

class gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, A, stride=1, residual=True):
        super().__init__()

        self.A = nn.Parameter(A + 0.0001*torch.ones(A.size()))

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

    def forward(self, x):
        res = self.residual(x)
        x, _ = self.gcn(x, self.A)
        x = self.bn(x) + res

        return self.relu(x)
