import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph


# gcn in self.gcn, remove [::-1]
class MFFNet2(nn.Module):
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

        self.conv_shift_1 = nn.Conv2d(3, 64, 1)
        self.gcn_shift_1 = gcn_o(64, 64, 3, A)
        self.tcn_shift = tcn(64, 128, 3, 1, 1)
        self.gcn_shift_2= gcn_o(128, 128, 3, A)
        self.conv_shift_2 = nn.Conv2d(128, 3, 1)

        self.tcn_motion_in = tcn(in_channels, 64, 3, 1, 1)
        self.tcn_pos_in_1 = tcn(in_channels, 64, 3, 1, 1)
        self.tcn_pos_in_2 = tcn(in_channels, 64, 3, 1, 1)
        self.conv_fusion_in = nn.Conv2d(128, 64, 1)

        self.tcn = nn.ModuleList((
            tcn(128, 64, 3, 1, 1),
            tcn(128, 64, 3, 2, 1),
	    gcn_o(128, 64, 3, A),
            tcn(128, 64, 3, 1, 1),
            tcn(128, 128, 3, 3, 1)
        ))

        self.tcn_back = nn.ModuleList((
            tcn(64, 64, 3, 1, 1),
            tcn(64, 64, 3, 1, 1),
            tcn(64, 64, 3, 2, 1),
            tcn(64, 64, 3, 1, 1),
            tcn(64, 64, 3, 1, 1)
        ))

        self.gcn = nn.ModuleList((
            gcn_o(64, 64, 3, A),
            gcn_o(64, 64, 3, A),
            gcn_o(64, 64, 3, A),
            gcn_o(64, 64, 3, A),
            gcn_o(64, 64, 3, A)
        ))

        self.tcn_end = tcn(128, 256, 9, 5, 1)
        self.gcn_end = gcn(256, 256)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def shift_adjust(self, x):
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

        x = self.shift_adjust(x)

        motion = torch.zeros_like(x)
        motion[:,:,1:,:] = x[:,:,1:,:] - x[:,:,:-1,:]

        # forwad
        motion = self.tcn_motion_in(motion)
        x_ = self.tcn_pos_in_1(x)
        x = self.tcn_pos_in_2(x)
        x = self.conv_fusion_in(torch.cat([x, motion], 1))

        for gcn, tcn_b, tcn in zip(self.gcn, self.tcn_back, self.tcn):
            x_ = tcn_b(gcn(x_))
            x = tcn(torch.cat([x_, x], 1))

        x = self.tcn_end(x)
        x = self.gcn_end(x)

        x = F.max_pool2d(x, x.size()[2:])
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

class gcn_o(nn.Module):
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

class gcn(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=False, coff_embedding=2, num_subset=3):
        super().__init__()


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

        self.num_subset = num_subset

        inter_channels = out_channels//coff_embedding
        self.inter_c = inter_channels

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.soft = nn.Softmax(-2)

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.residual(x)
        return y

class gtcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, residual=True):
        super(gtcn, self).__init__()
        self.gcn = gcn(in_channels, out_channels)
        self.tcn = tcn(out_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = tcn(in_channels, out_channels, kernel_size, stride)

    def forward(self, x, A):
        x = self.tcn(self.gcn(x, A)) + self.residual(x)
        return self.relu(x)
