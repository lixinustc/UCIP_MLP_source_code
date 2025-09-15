import torch
import torch.nn as nn

import math
from torch.nn import init
import torch.optim as optim
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
try:
    from timm.layers import to_2tuple
except ImportError:
    # 老版本 fallback
    from timm.models.layers.helpers import to_2tuple
from timm.models.vision_transformer import _cfg

# from utils import dict_to_string
from einops import rearrange


class CALayer(nn.Module):
    def __init__(self, channel, prompt_channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.mlp_du = nn.Sequential(Mlp(prompt_channel, prompt_channel // reduction, channel), nn.Sigmoid())

    def forward(self, x, prompt):
        '''
        x, prompt: [B, H, W, C]
        '''
        prompt = self.avg_pool(prompt.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        attn = self.mlp_du(prompt)
        return attn * x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x 

class SPADE(nn.Module):
    def __init__(self, p_dim, norm_nc, label_nc):
        super().__init__()
        
        ks = 3  ## spadeinstance5x5
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = p_dim // 2

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, prompt):

        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(prompt)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out


class ATMOp(nn.Module):
    def __init__(
            self, in_chans, out_chans, stride: int = 1, padding: int = 0, dilation: int = 1,
            bias: bool = True, dimension: str = ''
    ):
        super(ATMOp, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.dimension = dimension

        self.weight = nn.Parameter(torch.empty(out_chans, in_chans, 1, 1))  # kernel_size = (1, 1)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_chans))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        """
        ATM along one dimension, the shape will not be changed
        input: [B, C, H, W]
        offset: [B, C, H, W]
        """
        B, C, H, W = input.size()
        offset_t = torch.zeros(B, 2 * C, H, W, dtype=input.dtype, layout=input.layout, device=input.device)
        if self.dimension == 'w':
            offset_t[:, 1::2, :, :] += offset
        elif self.dimension == 'h':
            offset_t[:, 0::2, :, :] += offset
        else:
            raise NotImplementedError(f"{self.dimension} dimension not implemented")
        return deform_conv2d(
            input, offset_t, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation
        )

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'dimension={dimension}'
        s += ', in_chans={in_chans}'
        s += ', out_chans={out_chans}'
        s += ', stride={stride}'
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class ATMLayer(nn.Module):
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.dim = dim

        # self.atm_c = nn.Linear(dim, dim, bias=False)
        self.atm_h = ATMOp(dim, dim, dimension='h')
        self.atm_w = ATMOp(dim, dim, dimension='w')
        self.atm_local = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim) 

        self.fusion = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, offset):
        """
        x: [B, H, W, C]
        offsets: [B, 2C, H, W]
        """
        B, H, W, C = x.shape
        assert offset.shape == (B, 2 * C, H, W), f"offset shape not match, got {offset.shape}"
        w = self.atm_w(x.permute(0, 3, 1, 2), offset[:, :C, :, :].contiguous()).permute(0, 2, 3, 1).contiguous()
        h = self.atm_h(x.permute(0, 3, 1, 2), offset[:, C:, :, :].contiguous()).permute(0, 2, 3, 1).contiguous()
        # c = self.atm_c(x)
        # print("======x_local", x.size())
        c = self.atm_local(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        a = (w + h + c).permute(0, 3, 1, 2).contiguous().flatten(2).mean(2)
        
        a = self.fusion(a).reshape(B, C, 3).permute(2, 0, 1).contiguous().softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = w * a[0] + h * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + ' ('
        s += 'dim: {dim}'
        s += ')'
        return s.format(**self.__dict__)
    

class Dyanmic_Prompt_Basis(nn.Module):
    def __init__(self, dim, num_task=6, num_basis=8, embed_dim=128):
        super().__init__()
        self.basis_mlp = Mlp(dim, hidden_features=num_basis, out_features=num_basis)
        self.num_basis = num_basis
        self.dim = dim

        self.prompt = nn.Parameter(torch.rand(num_task, num_basis, 1, 1, embed_dim))
       
    
    def forward(self, x, task_id):
        '''
        x: image features [B, H, W, C]
        return: prompt [N, 1, H, W, C]  N is the number os tasks
        '''

        x_basis = self.basis_mlp(x)
        x_basis = rearrange(x_basis, 'b h w nb -> b nb h w').contiguous().unsqueeze(-1)  # B, nb, H, W, 1
        x_basis = F.softmax(x_basis, dim=1)
        prompts = self.prompt[task_id].contiguous()  # nt, nb, 1, 1, C -> B, nb, 1, 1, C
        # prompts = self.prompt.unsqueeze(0).repeat((b, 1, 1, 1, 1))  # B, nb, 1, 1, C
        prompts = prompts * x_basis  # B, nb, H, W, C
        prompts = prompts.sum(dim=1)  # B, H, W, C
        return prompts


class DynamicActiveBlock(nn.Module):
    def __init__(self, dim, p_dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 share_dim=1, downsample=None, new_offset=False,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.atm = ATMLayer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.downsample = downsample

        self.new_offset = new_offset
        self.share_dim = share_dim

        if new_offset:
            self.offset_layer = nn.Sequential(
                    norm_layer(dim + p_dim),
                    nn.Linear(dim + p_dim, dim * 2 // self.share_dim)
                )
        else:
            self.offset_layer = None

        self.spade = SPADE(p_dim, dim, p_dim)

    def forward(self, x, prompt_x, offset=None, task_id=None):
        """
        :param x: [B, H, W, C]
        :param offset: [B, 2C, H, W]
        """
        if self.offset_layer and offset is None:
            x_p = torch.cat([x, prompt_x], dim=-1)
            offset = self.offset_layer(x_p).repeat_interleave(self.share_dim, dim=-1).permute(0, 3, 1, 2).contiguous()  # [B, H, W, 2C/S] -> [B, 2C, H, W]
    
        
        x = x + self.drop_path(self.atm(self.norm1(x), offset))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.downsample is not None:
            x = self.downsample(x)

        x = self.spade(x.permute(0, 3, 1, 2).contiguous(), prompt_x.permute(0, 3, 1, 2).contiguous())
        # x = self.spade(x, prompt_x)
        
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.offset_layer:
            return x, offset
        else:
            return x

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + ' ('
        # s += 'new_offset: {offset}'
        s += ', share_dim: {share_dim}'
        s += ')'
        return s.format(**self.__dict__)


class Downsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Downsample_IR(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        # self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        # x = x.permute(0, 3, 1, 2)
        # x = self.proj(x)
        # x = x.permute(0, 2, 3, 1)
        return x



class PEG(nn.Module):
    """
    PEG
    from https://arxiv.org/abs/2102.10882
    """
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PEG, self).__init__()
        # depth conv
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=stride, padding=1, bias=True, groups=embed_dim)
        self.stride = stride

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x_conv = x
        x_conv = x_conv.permute(0, 3, 1, 2).contiguous()
        if self.stride == 1:
            x = self.proj(x_conv) + x_conv
        else:
            x = self.proj(x_conv)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Overlaped patch embedding, implemeted with 2D conv
    """
    def __init__(self, patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=64):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x = self.proj(x)
        return x


class OverlapPatchEmbed_IR(nn.Module):
    """
    Overlaped patch embedding, implemeted with 2D conv
    """
    def __init__(self, patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=64):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        x = self.proj(x)
        return x


class UCIP_MLP(nn.Module):
    """
    ActiveMLP
    https://arxiv.org/abs/2203.06108
    """
    def __init__(
            self,
            img_size=224,
            patch_size=5,
            in_chans=3,
            p_dim=32,
            depths=[2, 2, 4, 2],
            embed_dims=[64, 64, 64, 64],
            mlp_ratios=[4, 4, 4, 4],
            share_dims=[1, 1, 1, 1],  # how many channels share one offset
            drop_path_rate=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            intv=2,  # interval for generating new offset
            **kwargs
    ):

        super().__init__()

        self.depths = depths
        self.intv = intv
        self.patch_embed = OverlapPatchEmbed_IR(patch_size=5, stride=1, padding=2, in_chans=3, embed_dim=embed_dims[0])


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        ii = 0
        self.blocks = nn.ModuleList()
        self.prompts = nn.ModuleList()
        for i in range(len(depths)):
            _block = nn.ModuleList([
                DynamicActiveBlock(embed_dims[i],
                            p_dim=p_dim,
                            mlp_ratio=mlp_ratios[i],
                            drop_path=dpr[ii + j],
                            share_dim=share_dims[i],
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                            downsample = None,
                            new_offset=(j % self.intv == 0 and j != depths[i] - 1),
                            ) for j in range(depths[i])
            ])
            _prompt = Dyanmic_Prompt_Basis(embed_dims[i], 6, 8, p_dim)
            self.blocks.append(_block)
            self.prompts.append(_prompt)
            ii += depths[i]

        # PEG for each resolution feature map
        self.pos_blocks = nn.ModuleList(
            [PEG(ed, ed) for ed in embed_dims]
        )

        self.norm = norm_layer(embed_dims[-1])

        num_feat = 64
        self.conv_after_body = nn.Conv2d(embed_dims[-1], embed_dims[-1], 3, 1, 1)
        self.conv_before_up = nn.Sequential(nn.Conv2d(embed_dims[-1], num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upscale = 4
        if self.upscale == 4:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, ATMOp):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(['pos_blocks.' + n for n, p in self.pos_blocks.named_parameters()])

    def forward_blocks(self, x, task_id):
        for i in range(len(self.depths)):
            res = x
            prompt = self.prompts[i](x, task_id)
            for j, blk in enumerate(self.blocks[i]):
                if j % self.intv == 0 and j != len(self.blocks[i]) - 1:
                    x = self.pos_blocks[i](x)
                    x, offset = blk(x, prompt, None, task_id=task_id)
                else:
                    x = blk(x, prompt, offset, task_id=task_id)

            x = x + res
        return x

    def forward(self, x, task_id=[0]):
        """
        x: [B, 3, H, W]
        """
        x = self.patch_embed(x)
        res = x
        x = x.permute(0, 2, 3, 1).contiguous() 

        x = self.forward_blocks(x, task_id)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.conv_after_body(x) + res
        x = self.conv_before_up(x)
        x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
        if self.upscale == 4:
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        return x

def UCIP_Middle(p_dim=64, pretrained=False, img_size=None, **kwargs):
    depths = [6, 6, 6, 6, 6, 6]
    mlp_ratios = [4, 4, 4, 4, 4, 4]
    embed_dims = [128, 128, 128, 128, 128, 128]
    share_dims = [2, 2, 2, 2, 2, 2]
    model = UCIP_MLP(img_size=img_size, depths=depths, p_dim=p_dim, embed_dims=embed_dims, mlp_ratios=mlp_ratios, share_dims=share_dims, intv=2, **kwargs)
    model.default_cfg = _cfg
    return model


