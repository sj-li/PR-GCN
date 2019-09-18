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
            AttentionGenerator(),
            AttentionGenerator(),
            AttentionGenerator()
        ))

        self.fusion = nn.ModuleList((
            MFFModule(64, 64),
            MFFModule(64, 64),
            MFFModule(128, 128)
        ))

        self.block_num = len(self.fusion)

    def BasicBlock(self, pos_feat, tmf_feat, smf_feat, atten_p, block_num):
        tmf_feat = self.GCN_tmf[block_num](tmf_feat, self.A*atten_p)
        smf_feat = self.GCN_smf[block_num](smf_feat, self.A*atten_p)

        atten = self.AttenGen[block_num](tmf_feat, smf_feat)

        pos_feat = self.GCN_pos[block_num](pos_feat, self.A*atten)

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
        tmf[:,:,:-1,:,:] = pos[:,:,1:,:,:] - pos[:,:,:-1,:,:]
        tmf[:,:,-1,:,:] = tmf[:,:,-2,:,:]
        tmf_feat = tmf
        # TODO: normalization

        # build Spatial Movement Field
        smf = torch.zeros_like(pos)
        center = pos[:,:,:,self.graph.center].unsqueeze(-1).expand(V)
        smf = pos - center
        smf_feat = smf
        # TODO: normalization

        atten = nn.Identity(A.size(0))

        for i in range(self.block_num):
            pos_feat, tmf_feat, smf_feat, atten = self.BasicBlock(pos_feat, tmf_feat, smf_feat, atten)

        # global pooling
        feat = F.avg_pool2d(feat, feat.size()[2:])
        feat = feat.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        feat = self.fcn(feat)
        feat = feat.view(feat.size(0), -1)

        return feat

        

class AttentionGenerator(nn.Module):
    def __init__(self):
        pass

    def forward(self, smf_feat, tfm_feat):
        pass

class MFFModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        
        self.conv_fusion = nn.Conv2d(in_channels*2,
                                     out_channels)
    
    def forward(self, pos_feat, tfm_feat, A):
        feat = torch.cat((pos_feat, tfm_feat), 1)
        feat = self.conv_fusion(feat)
        return feat
        
        
class GCNModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias = True,
                 residual = True):
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              bias=bias)

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
    
    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        res = self.residual(x)

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        x = x.contiguous()

        x = x + res

        return x, A

if __name__ == '__main__':
    model = MFFNet()
  