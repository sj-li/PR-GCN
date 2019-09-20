import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mmskeleton.ops.st_gcn import ConvTemporalGraphical, Graph

class MFFNet(nn.Module):
    """Movemend Field Fusion Network for Skeleton-based Action Recognition
    
    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 **kwargs):

        super().__init__()
        
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                 dtype=torch.float32,
                 requires_grad=False)
        A.squeeze_()
        
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        self.GCN_pos = nn.ModuleList((
            GCNModule(in_channels, 64),
            GCNModule(64, 64),
            GCNModule(64, 128)
        ))

        self.GCN_tmf = nn.ModuleList((
            GCNModule(in_channels, 64),
            GCNModule(64, 64),
            GCNModule(64, 128)
        ))

        self.TCN = nn.ModuleList((
            TCNModule(64, 3, 1, 1),
            TCNModule(64, 3, 1, 2),
            TCNModule(128, 3, 1, 3)
        ))
        
        self.AttenGen = nn.ModuleList((
            AttentionGenerator(64),
            AttentionGenerator(64),
            AttentionGenerator(128)
        ))

        self.fusion = nn.ModuleList((
            MFFModule(64, 64),
            MFFModule(64, 64),
            MFFModule(128, 128)
        ))

        self.block_num = len(self.fusion)

        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def BasicBlock(self, pos_feat, tmf_feat, atten_p, block_num):
        tmf_feat, _ = self.GCN_tmf[block_num](tmf_feat, atten_p)

        atten = self.AttenGen[block_num](tmf_feat, self.A)

        pos_feat, _ = self.GCN_pos[block_num](pos_feat, atten)
        pos_feat = self.fusion[block_num](pos_feat, tmf_feat)

        tmf_feat = self.TCN[block_num](tmf_feat)

        return pos_feat, tmf_feat, atten

    def normalize(self, feat):
        N, C, T, V, M = feat.size()
        feat = feat.permute(0, 4, 3, 1, 2).contiguous()
        feat = feat.view(N * M, V * C, T)
        feat = self.data_bn(feat)
        feat = feat.view(N, M, V, C, T)
        feat = feat.permute(0, 1, 3, 4, 2).contiguous()
        feat = feat.view(N * M, C, T, V)

        return feat

    def forward(self, pos):

        # data normalization
        N, C, T, V, M = pos.size()

        tmf = torch.zeros_like(pos)
        tmf[:,:,:-1,:,:] = pos[:,:,1:,:,:] - pos[:,:,:-1,:,:]
        tmf[:,:,-1,:,:] = tmf[:,:,-2,:,:]

        pos_feat = self.normalize(pos)
        tmf_feat = self.normalize(tmf)

        atten_p = self.A.unsqueeze(0).unsqueeze(0).repeat(N*M, T, 1, 1)

        for i in range(self.block_num):
            pos_feat, tmf_feat, atten = self.BasicBlock(pos_feat, tmf_feat, atten_p, i)

        # global pooling
        feat = F.avg_pool2d(pos_feat, pos_feat.size()[2:])
        feat = feat.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        feat = self.fcn(feat)
        feat = feat.view(feat.size(0), -1)

        return feat

        

class AttentionGenerator(nn.Module):
    def __init__(self, in_channels, alpha=0.1):
        super().__init__()
        self.alpha = alpha

        self.conv_fusion = nn.Conv2d(in_channels, 64, kernel_size = 1)
        self.conv_atten = nn.Conv2d(128, 1, kernel_size = 1)
        self.activiation = nn.LeakyReLU(self.alpha)
        

    def forward(self, feat, A):
        feat = self.conv_fusion(feat)

        B, C, T, V = feat.shape

        feat1 = feat.unsqueeze(1)
        feat1 = feat1.repeat(1, 18, 1, 1, 1)
        feat1 = feat1.permute(0, 2, 3, 4, 1).contiguous()
        feat1 = feat1.view(B, C, T, V*V)

        feat2 = feat.unsqueeze(0)
        feat2 = feat2.repeat(18, 1, 1, 1, 1)
        feat2 = feat2.permute(1, 2, 3, 0, 4).contiguous()
        feat2 = feat2.view(B, C, T, V*V)

        feat = torch.cat([feat1, feat2], dim=1)
        feat = self.conv_atten(feat)
        feat = feat.squeeze_().view(B, T, V, V)

        feat = self.activiation(feat)
        feat_exp = torch.exp(feat)
        feat_exp = feat_exp*A

        feat_agg = torch.sum(feat_exp, dim=-1).unsqueeze_(-1).repeat(1, 1, 1, V)
        atten = feat_exp/feat_agg

        return atten

class MFFModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        
        self.conv_fusion = nn.Conv2d(in_channels*2,
                                     out_channels,
                                     kernel_size=1)
    
    def forward(self, pos_feat, tfm_feat):
        feat = torch.cat((pos_feat, tfm_feat), 1)
        feat = self.conv_fusion(feat)
        return feat

class GCNModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()


        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              bias=True)


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

    def forward(self, x, atten):

        res = self.residual(x)

        x = self.conv(x)
        x = x + res

        x = x.permute(0, 2, 3, 1)
        x = torch.matmul(atten, x)
        x = x.permute(0, 3, 1, 2)
        

        return self.relu(x), atten

class TCNModule(nn.Module):
    def __init__(self,
                 in_channels,
                 t_kernel_size=1,
                 t_stride=1,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        # t_padding = ((t_kernel_size - 1) // 2, 0)
        t_padding = ((t_dilation*(t_kernel_size-1))//2, 0)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(t_kernel_size, 1),
                stride=(t_stride, 1),
                padding=t_padding,
                dilation=(t_dilation, 1),
                bias=bias
            ),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, feat):
        feat = self.tcn(feat)
        return feat
