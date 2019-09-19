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
        self.register_buffer('A', A)

        kernel_size = A.size(0)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        self.GCN_pos = nn.ModuleList((
            GCNModule(in_channels, 64, kernel_size),
            GCNModule(64, 64, kernel_size),
            GCNModule(64, 128, kernel_size)
        ))

        self.GCN_tmf = nn.ModuleList((
            GCNModule(in_channels, 64, kernel_size),
            GCNModule(64, 64, kernel_size),
            GCNModule(64, 128, kernel_size)
        ))
        
        self.GCN_smf = nn.ModuleList((
            GCNModule(in_channels, 64, kernel_size),
            GCNModule(64, 64, kernel_size),
            GCNModule(64, 128, kernel_size)
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

    def BasicBlock(self, pos_feat, tmf_feat, smf_feat, atten_p, block_num):
        tmf_feat = self.GCN_tmf[block_num](tmf_feat, atten_p)
        smf_feat = self.GCN_smf[block_num](smf_feat, atten_p)

        atten = self.AttenGen[block_num](tmf_feat, smf_feat, self.A)

        pos_feat = self.GCN_pos[block_num](pos_feat, atten)

        return pos_feat, tmf_feat, smf_feat, atten

    def forward(self, pos):

        # data normalization
        N, C, T, V, M = pos.size()
        pos = pos.permute(0, 4, 3, 1, 2).contiguous()
        pos = pos.view(N * M, V * C, T)
        pos = self.data_bn(pos)
        pos = pos.view(N, M, V, C, T)
        pos = pos.permute(0, 1, 3, 4, 2).contiguous()
        pos = pos.view(N * M, C, T, V)
        pos_feat = pos

        # build Temporal Movement Field
        tmf = torch.zeros_like(pos)
        tmf[:,:,:-1,:] = pos[:,:,1:,:] - pos[:,:,:-1,:]
        tmf[:,:,-1,:] = tmf[:,:,-2,:]
        tmf_feat = tmf
        # TODO: normalization

        exit()

        # build Spatial Movement Field
        smf = torch.zeros_like(pos)
        center = pos[:,:,:,self.graph.center].unsqueeze(-1).repeat(1, 1, 1, V)
        smf = pos - center
        smf_feat = smf
        # TODO: normalization

        atten = nn.Identity(A.size(0))

        for i in range(self.block_num):
            pos_feat, tmf_feat, smf_feat, atten = self.BasicBlock(pos_feat, tmf_feat, smf_feat, atten)

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

        self.conv_fusion = nn.Conv2d(in_channels*2, 64, kernel_size = 1)
        self.conv_atten = nn.Conv2d(128, 1, kernel_size = 1)
        self.activiation = nn.LeakyReLU(self.alpha)
        

    def forward(self, smf_feat, tfm_feat, A):
        feat = torch.cat((smf_feat, tfm_feat), 1)
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
    
    def forward(self, pos_feat, tfm_feat, A):
        feat = torch.cat((pos_feat, tfm_feat), 1)
        feat = self.conv_fusion(feat)
        return feat

class GCNModule(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()


        self.kernel_size = kernel_size
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
        x = atten*x
        x = x.permute(0, 3, 1, 2)
        

        return self.relu(x), atten