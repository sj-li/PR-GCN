import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph


# gcn in self.gcn, remove [::-1]
class CST_GCN_2(nn.Module):
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

        # self.conv_shift_1 = nn.Conv2d(3, 64, 1)
        # self.gcn_shift_1 = gcn_o(64, 64, 3, A)
        # self.tcn_shift = tcn(64, 128, 3, 1, 1)
        # self.gcn_shift_2= gcn_o(128, 128, 3, A)
        # self.conv_shift_2 = nn.Conv2d(128, 3, 1)

        self.tcn_pos_in = tcn(in_channels, 64, 3, 1, 1)
        self.tcn_motion_in = tcn(in_channels, 64, 3, 1, 1)

        self.tcn = nn.ModuleList((
            tcn(64, 64, 3, 1, 1),
            tcn(64, 64, 3, 2, 1),
	        gcn(64, 64, A, add_atten=True),
            tcn(64, 64, 3, 1, 1),
            tcn(64, 64, 3, 3, 1)
        ))

        self.gcn = nn.ModuleList((
            gcn(128, 64, A, add_atten=True),
            gcn(128, 64, A, add_atten=True),
            gcn(128, 64, A, add_atten=True),
            gcn(128, 64, A, add_atten=True),
            gcn(128, 128, A, add_atten=True)
        ))

        self.tcn_end = tcn(128, 256, 9, 5, 1)
        self.gcn_end = gcn(256, 256, A, only_atten=True)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    # def shift_adjust(self, x):
    #     shift = self.conv_shift_1(x)
    #     shift = self.gcn_shift_1(shift)
    #     shift = self.tcn_shift(shift)
    #     shift = self.gcn_shift_2(shift)
    #     shift = self.conv_shift_2(shift)
    #     shift[:, 2, :, :] = 0

    #     x = x + shift

    #     return x

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # x = self.shift_adjust(x)

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

class gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, num_subset=3, only_atten=False, add_atten=False, coff_embedding=2):
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

        assert num_subset == A.size(0)
        self.num_subset = num_subset
        self.A = nn.Parameter(A + 0.0001*torch.ones(A.size())) if not only_atten else torch.zeros_like(A)

        if only_atten == True:
            add_atten = True

        self.add_atten = add_atten
        if add_atten == True:
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
            A1 = self.A[i].cuda(x.get_device())
            if self.add_atten == True:
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = A1 + self.A[i].cuda(x.get_device())
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.residual(x)
        return y
