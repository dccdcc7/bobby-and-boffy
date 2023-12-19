import torch
import torch.nn as nn
import math


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        # 初始化
        super(se_block, self).__init__()
        # 自适应平均池化的输出高宽设置为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #定义序列模型
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        # torch.size函数：从左往右依次是batchsize，channel，height，weight
        b, c, _, _ = x.size()
        # [b,c,h,w]->[b,c,1,1]->[b,c] 因为全连接层的输入与输出一般为二维张量[batchsize，size]
        avg = self.avg_pool(x).view(b,c)
        # [b,c]->[b,c/ratio]->[b,c]->[b,c，1,1]  把全连接层输出的宽高维度1×1还原
        avg = self.fc(avg).view(b, c, 1, 1)
        return x * avg

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.fc = nn.Sequential(
        #     nn.Linear(in_planes, in_planes // ratio, False),
        #     nn.ReLU(),
        #     nn.Linear(in_planes // ratio, in_planes, False),
        # )

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, h, w = x.size()
        # avg_pool_out = self.avg_pool(x).view(b,c)
        # max_pool_out = self.max_pool(x).view(b,c)
        #
        # avg_fc_out = self.fc(avg_pool_out)
        # max_fc_out = self.fc(max_pool_out)
        #
        # out = avg_fc_out + max_fc_out
        # out = self.sigmoid(out)
        # return out
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = kernel_size // 2  向下取整
        padding = 3 if kernel_size == 7 else 1
        # 输入通道数，输出通道数，卷积核大小，步长（默认1）
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # torch.size函数：从左往右依次是batchsize，channel，height，weight
        b, c, h, w = x.size()
        # dim=1对应通道, keepdim=True表示把通道维度保留下来
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        pool_out = torch.cat([avg_pool_out, max_pool_out], dim=1)
        out = self.conv1(pool_out)
        return self.sigmoid(out)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio)
        self.spatialattention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 原特征层乘以权重
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        # 根据输入通道数自适应计算卷积核大小（输入通道数大的话，卷积核就大一点，反之则小一些）
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # avg = self.avg_pool(x)
        # y = self.conv(avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y = self.sigmoid(y)
        # return x * y.expand_as(x)

        # 一般来说，一维卷积nn.Conv1d用于文本数据，只对宽度卷积，不对宽度卷积
        # 通常输入大小为word_embedding_dim * max_length
        # 卷积核窗口在句子长度的方向上滑动，进行卷积操作
        avg = self.avg_pool(x).view(b,1,c)
        out = self.conv(avg)
        out = self.sigmoid(out).view(b,c,1,1)
        return x * out

class GC_Block(nn.Module):
    def __init__(self, inplanes, ratio=0.25, pooling_type='att', fusion_types=('channel_add', )):
        super(GC_Block, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class MSCAM(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(MSCAM, self).__init__()
        inter_channels = in_channels//ratio
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels, eps=0.001, momentum=0.03),
            SiLU(),
            DConv(inter_channels, inter_channels, kernel_size=3, padding=3, dilation=3),
            DConv(inter_channels, inter_channels, kernel_size=3, padding=3, dilation=3),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.03),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels, eps=0.001, momentum=0.03),
            SiLU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.03),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att_local = self.local_att(x)
        att_global = self.global_att(x)
        att = att_global + att_local
        weight = self.sigmoid(att)
        return weight