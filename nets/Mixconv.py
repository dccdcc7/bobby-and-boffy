import math
import torch
import torch.nn as nn

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DWConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ECA_weight(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_weight, self).__init__()
        # 根据输入通道数自适应计算卷积核大小（输入通道数大的话，卷积核就大一点，反之则小一些）
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 一般来说，一维卷积nn.Conv1d用于文本数据，只对宽度卷积，不对宽度卷积
        # 通常输入大小为word_embedding_dim * max_length
        # 卷积核窗口在句子长度的方向上滑动，进行卷积操作
        x = self.avg_pool(x).view(b, 1, c)
        x = self.conv(x)
        x = self.sigmoid(x).view(b, c, 1, 1)
        return x


class se_block(nn.Module):
    def __init__(self, channel, ratio=4):
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

#-------------------------------------------------#
#   MSFEM 多尺度特征提取模块
#-------------------------------------------------#
class MSFEM1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFEM1, self).__init__()
        self.in_channels = in_channels
        self.conv_dw1 = DWConv(in_channels//2, in_channels//2, 3)
        self.conv_dw2 = DWConv(in_channels//2, in_channels//2, 5)
        self.eca = ECA_weight(in_channels//2)
        self.split_channel = in_channels//2
        self.softmax = nn.Softmax(dim=1)
        self.conv_pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.act = SiLU()

    def forward(self, x):
        batch_size = x.shape[0]
        #   引入残差边
        res = x
        c = self.in_channels
        #   Channel Split
        x1 = torch.split(x, c//2, dim=1)[0]
        x2 = torch.split(x, c//2, dim=1)[1]
        #   Mixed Depth-wise convolution
        x1 = self.conv_dw1(x1)
        x2 = self.conv_dw2(x2)
        #   concat
        feats = torch.cat((x1, x2), dim=1)
        feats = feats.view(batch_size, 2, self.split_channel, feats.shape[2], feats.shape[3])
        #   不同通道内部的注意力分配
        x1_eca = self.eca(x1)
        x2_eca = self.eca(x2)
        #   获得整体的通道注意力并softmax进行归一化，   feat * attention
        x_eca = torch.cat((x1_eca, x2_eca), dim=1)
        attention_vectors = x_eca.view(batch_size, 2, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats = feats * attention_vectors
        for i in range(2):
            x_eca_weight_fp = feats[:, i, :, :]
            if i == 0:
                out = x_eca_weight_fp
            else:
                out = torch.cat((x_eca_weight_fp, out), 1)
        #   逐点卷积促进通道间信息交互
        out = self.conv_pw(out)
        #   残差连接
        out = self.act(out + res)
        return out

class MSFEM2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFEM2, self).__init__()
        self.in_channels = in_channels
        self.conv_dw1 = DWConv(in_channels//4, in_channels//4, 3)
        self.conv_dw2 = DWConv(in_channels//4, in_channels//4, 5)
        self.conv_dw3 = DWConv(in_channels//4, in_channels//4, 7)
        self.conv_dw4 = DWConv(in_channels//4, in_channels//4, 9)
        self.eca = ECA_weight(in_channels//4)
        self.split_channel = in_channels//4
        self.softmax = nn.Softmax(dim=1)
        self.conv_pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = SiLU()

    def forward(self, x):
        batch_size = x.shape[0]
        #   引入残差边
        res = x
        c = self.in_channels
        #   Channel Split
        x1 = torch.split(x, c//4, dim=1)[0]
        x2 = torch.split(x, c//4, dim=1)[1]
        x3 = torch.split(x, c//4, dim=1)[2]
        x4 = torch.split(x, c//4, dim=1)[3]
        #   Mixed Depth-wise convolution
        x1 = self.conv_dw1(x1)
        x2 = self.conv_dw2(x2)
        x3 = self.conv_dw3(x3)
        x4 = self.conv_dw4(x4)
        #   concat
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
        #   不同通道内部的注意力分配
        x1_eca = self.eca(x1)
        x2_eca = self.eca(x2)
        x3_eca = self.eca(x3)
        x4_eca = self.eca(x4)
        #   获得整体的通道注意力并softmax进行归一化，   feat * attention
        x_eca = torch.cat((x1_eca, x2_eca, x3_eca, x4_eca), dim=1)
        attention_vectors = x_eca.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats = feats * attention_vectors
        for i in range(4):
            x_eca_weight_fp = feats[:, i, :, :]
            if i == 0:
                out = x_eca_weight_fp
            else:
                out = torch.cat((x_eca_weight_fp, out), 1)
        #   逐点卷积促进通道间信息交互
        out = self.conv_pw(out)
        #   残差连接
        out = self.act(out + res)
        return out