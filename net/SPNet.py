import torch
import torch.nn as nn
import torch.nn.functional as F
from net.ResNet import resnet50
from math import log
from net.Res2Net import res2net50_v1b_26w_4s
import cv2
from mmcv.cnn import build_norm_layer
from net.pvtv2 import pvt_v2_b0,pvt_v2_b1,pvt_v2_b2,pvt_v2_b2_li,pvt_v2_b3,pvt_v2_b4,pvt_v2_b5


class ConvBNR(nn.Module): #一个卷积层  为了方便调用设计的   Conv + BatchNorm2d + ReLu
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):  #初始话方法  构造函数实例化时进行一些初始化工作
        super(ConvBNR, self).__init__()  #一个初始化的操作   self代表实例本身   就是说  当构建了一个对象之后  这个self就代表这个对象自己了

        self.block = nn.Sequential(  #将模块实例化操作
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
                     #输入通道数  输出通道数     核尺寸       步长             间隔               膨胀率
                # 所以现在的语法（确切的说是1x1的卷积）根本不用定义卷积核的大小，定义输出和输入的通道数即可，后台会自动的设置变化的每个层中的卷积核的大小
            nn.BatchNorm2d(planes), #BN  批量归一化层  再看看把 。。。淦
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  #self是格式问题  实例方法  前向传播。。。
        return self.block(x)  #二者的定义似乎都是重合的  前向传播可能定义了一个实例的过程


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)  #卷积层  就是赋值的一个操作   然后在类中创建了一个对象  算一种格式把
        self.bn = nn.BatchNorm2d(planes)  # bn层  批量归一化
        self.relu = nn.ReLU(inplace=True) #激活函数

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    #熟悉各种运算的尺寸变化的过程，进行操作  卷积  批量归一化 激活函数

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(64, 32)   # f2 conv1x1
        self.reduce4 = Conv1x1(512, 256) # f5 conv1x1
        # self.reduced = Conv1x1(512, 128)
        self.block = nn.Sequential(     #  与论文中描述有所不同
            ConvBNR(256 + 32, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))  #conv卷积   conv1d 一维卷积核   conv2d （二维卷积核） 二维图像卷积  第三个参数为卷积核的大小

    def forward(self, x4, x1):
        size = x1.size()[2:]  #获取size 返回的 torch.Size 对象的第三个维度及其之后的所有维度大小
        x1 = self.reduce1(x1) # f2 conv1x1  卷积后的结果   （104x104x256） --> （104x104x64）
        x4 = self.reduce4(x4) # f5 conv1x1  卷积后的结果    （13x13x2048）-->  (13x13x256)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)  #采样的函数 根据size来进行上采样或者下采样
        # depths=F.interpolate(depths, size, mode='bilinear', align_corners=False)
        # d2 = self.reduced(d2)
        # d2 = F.interpolate(d2, size, mode='bilinear', align_corners=False)

        out = torch.cat((x4, x1), dim=1) #连接操作
     #就纯纯的连接  把两个东西对接起来   后面必定会有一个卷积把他卷到一起
        # out=out*depths*0.4+out*0.6
        out = self.block(out)  #
        return out  #输出 fe  即边缘特征  由于后续的指导训练


# class EAM_depth(nn.Module):
#     def __init__(self):
#         super(EAM_depth, self).__init__()
#         self.reduce1 = Conv1x1(256, 64)   # f2 conv1x1
#         self.reduce4 = Conv1x1(2048, 256) # f5 conv1x1
#         self.reduced = Conv1x1(512, 128)
#         self.block = nn.Sequential(     #  与论文中描述有所不同
#             ConvBNR(256 + 64, 256, 3),
#             ConvBNR(256, 256, 3),
#             nn.Conv2d(256, 1, 1))  #conv卷积   conv1d 一维卷积核   conv2d （二维卷积核） 二维图像卷积  第三个参数为卷积核的大小
#
#         self.block1 = nn.Sequential(  # 与论文中描述有所不同
#         ConvBNR(2048, 256, 3),
#         ConvBNR(256, 256, 3),
#         nn.Conv2d(256, 1, 1))
#
#         self.block2 = nn.Sequential(  # 与论文中描述有所不同
#             ConvBNR(1024, 256, 3),
#             ConvBNR(256, 256, 3),
#             nn.Conv2d(256, 1, 1))
#
#         self.block3 = nn.Sequential(  # 与论文中描述有所不同
#             ConvBNR(512, 256, 3),
#             ConvBNR(256, 256, 3),
#             nn.Conv2d(256, 1, 1))
#
#         self.block4 = nn.Sequential(  # 与论文中描述有所不同
#             ConvBNR(256, 256, 3),
#             ConvBNR(256, 256, 3),
#             nn.Conv2d(256, 1, 1))
#
#     def forward(self, x4,x3,x2, x1,d4,d3,d2,d1):
#         size = x4.size()[2:]  #获取size 返回的 torch.Size 对象的第三个维度及其之后的所有维度大小
#      #    x1 = self.reduce1(x1) # f2 conv1x1  卷积后的结果   （104x104x256） --> （104x104x64）
#      #    x4 = self.reduce4(x4) # f5 conv1x1  卷积后的结果    （13x13x2048）-->  (13x13x256)
#      #    x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)  #采样的函数 根据size来进行上采样或者下采样
#      #    # depths=F.interpolate(depths, size, mode='bilinear', align_corners=False)
#      #    # d2 = self.reduced(d2)
#      #    # d2 = F.interpolate(d2, size, mode='bilinear', align_corners=False)
#      #
#      #    out = torch.cat((x4, x1), dim=1) #连接操作
#      # #就纯纯的连接  把两个东西对接起来   后面必定会有一个卷积把他卷到一起
#      #    # out=out*depths*0.4+out*0.6
#      #    out = self.block(out)  #
#
#         t1 = d1+x1
#         t1 = F.interpolate(t1, size, mode='bilinear', align_corners=False)
#         edge1=self.block1(t1)
#
#         t2=(d2+x2)
#         t2 = F.interpolate(t2, size, mode='bilinear', align_corners=False)
#         t2=t2*edge1+t2
#         edge2 = self.block2(t2)
#
#         t3 = (d3 + x3)
#         t3 = F.interpolate(t3, size, mode='bilinear', align_corners=False)
#         t3=t3*edge2+t3
#         edge3 = self.block3(t3)
#
#         t4 = (d4 + x4)
#         t4 = F.interpolate(t4, size, mode='bilinear', align_corners=False)
#         t4=t4*edge3+t4
#         edge4 = self.block4(t4)
#
#         return edge1,edge4  #输出 fe  即边缘特征  由于后续的指导训练
#




class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1  #    t % 2 等价于if(t%2!=0)
        self.conv2d = ConvBNR(channel, channel, 3)  #卷积核位3x3

        self.refine = Conv1x1(channel*2, channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1) #平均池化层  GAP
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att,dep): #att --》 fe    c-->fi c特征  att边界
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)

        # dep=F.interpolate(dep, c.size()[2:], mode='bilinear', align_corners=False)

        # x = c * att + c + dep*0.5 #逐位相乘 再相加

        #
        x = torch.cat((c*att+c,dep*c), dim=1)
        x=self.refine(x)

        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
         # squeeze(-1) 作用：从数组的形状中删除单维度条目，即把shape中为1的维度去掉 哪个维度数值为1 就删除那个维度  -1和1 效果一样  降维
        # transpose（）表示转置
        wei = self.sigmoid(wei)  #获得通道关注度
        x = x * wei  #  x和wei元素逐位相乘  把每个通道的数据和权重相乘
                      #少一个1x1的卷积啊  最后添加的有
        return x,dep


class EFM2(nn.Module):
    def __init__(self, channel):
        super(EFM2, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1  #    t % 2 等价于if(t%2!=0)
        self.conv2d = ConvBNR(channel, channel, 3)  #卷积核位3x3

        self.refine = Conv1x1(channel*2, channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1) #平均池化层  GAP
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att): #att --》 fe    c-->fi c特征  att边界
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)

        # dep=F.interpolate(dep, c.size()[2:], mode='bilinear', align_corners=False)

        x = c * att + c # 元素逐位相乘 再相加
        # temp=c+dep*att
        # x = torch.cat((c*att,temp), dim=1)
        # x=self.refine(x)

        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
         # squeeze(-1) 作用：从数组的形状中删除单维度条目，即把shape中为1的维度去掉 哪个维度数值为1 就删除那个维度  -1和1 效果一样  降维
        # transpose（）表示转置
        wei = self.sigmoid(wei)  #获得通道关注度
        x = x * wei  #  x和wei元素逐位相乘  把每个通道的数据和权重相乘
                      #少一个1x1的卷积啊  最后添加的有
        return x





class CAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, lf, hf): # lf 低级特征   hf高级特征
        if lf.size()[2:] != hf.size()[2:]:  #如果二者的通道数不一样  就进行采样保持一致
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)  #连接后卷积
        xc = torch.chunk(x, 4, dim=1) #按照 第二维度 对张量进行切分   然后再处理每一部分的张量  这也是本文的创新点  处理通道间的内容
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)   #对每个通道内的张量进行操作
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)#  连接  相加 最后卷积

        return x





class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1):
        super(CBR, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        return x


class CBR2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1):
        super(CBR2, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)

    def forward(self, x, x2):
        size = x.size()[2:]  # 获取size 返回的 torch.Size 对象的第三个维度及其之后的所有维度大小

        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)  # 采样的函数 根据size来进行上采样或者下采样

        x=x+x2*x

        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        return x





class Net(nn.Module):
    def __init__(self,fun_str = 'pvt_v2_b3'):
        super(Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)  #调用resnet网络  获取到初始的特征
        self.pvtb, embedding_dims = eval(fun_str)()
        # if self.training:
        # self.initialize_weights()
        self.eam = EAM()

        # self.eam_depth = EAM_depth()

        self.efm1 = EFM(64)
        self.efm2 = EFM(128)
        self.efm3 = EFM(320)
        self.efm4 = EFM(512)

        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 128)
        self.reduce3 = Conv1x1(320, 256)
        self.reduce4 = Conv1x1(512, 256)

        self.cam1 = CAM(128, 64)
        self.cam2 = CAM(256, 128)
        self.cam3 = CAM(256, 256)


        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(128, 1, 1)
        self.predictor3 = nn.Conv2d(256, 1, 1)


        self.predict_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1))

        self.cbr1 = CBR(in_channels=256, out_channels=64,
                                          kernel_size=3, stride=1,
                                          dilation=1, padding=1)


        self.predict_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1, stride=1))

        self.cbr2 = CBR2(in_channels=512, out_channels=128,
                                          kernel_size=3, stride=1,
                                          dilation=1, padding=1)


        self.predict_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1, stride=1))

        self.cbr3 = CBR2(in_channels=1024, out_channels=256,
                                          kernel_size=3, stride=1,
                                          dilation=1, padding=1)


        self.predict_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1, stride=1))

        self.cbr4 = CBR2(in_channels=2048, out_channels=512,
                                          kernel_size=3, stride=1,
                                          dilation=1, padding=1)





    # def initialize_weights(self):
    # model_state = torch.load('./models/resnet50-19c8e357.pth')
    # self.resnet.load_state_dict(model_state, strict=False)

    def forward(self, x, depth):

        # x1, x2, x3, x4 = self.resnet(x) # 特征获取 取 2-5层特征
        x1,x2,x3,x4=self.pvtb(x)

        d1, d2, d3, d4 = self.resnet(depth)  # 特征获取 取 2-5层特征

        edge = self.eam(x4, x1)  # f2 + f5 -》 eam --》get  edge

        d1=self.cbr1(d1)
        d1=self.predict_conv1(d1)

        d2=self.cbr2(d2,d1)
        d2=self.predict_conv2(d2)

        d3 = self.cbr3(d3,d2)
        d3 = self.predict_conv3(d3)

        d4 = self.cbr4(d4,d3)
        d4 = self.predict_conv4(d4)



        edge== torch.sigmoid(edge)


        x1a, dep1 = self.efm1(x1, edge, d1)


        x2a,dep2 = self.efm2(x2, edge,d2)        # fi+edge_att  ---- EFM---》 fa_i

        x3a,dep3 = self.efm3(x3, edge,d3)

        x4a,dep4 = self.efm4(x4, edge,d4)



        x1r = self.reduce1(x1a)  #对上面的efm补的 一个1x1的卷积层

        x2r = self.reduce2(x2a)

        x3r = self.reduce3(x3a)

        x4r = self.reduce4(x4a)

        x34 = self.cam3(x3r, x4r)     # 每次的累加处理  3+4

        x234 = self.cam2(x2r, x34)    #  3+4+2

        x1234 = self.cam1(x1r, x234)  #3+4+2+1

       #在做出了一堆的特征后，就把他的通道维度卷成1  就是黑白的好辩认  至于区分这个点应该是黑还是白 就是模型的作用了  根据训练和数据集来判定

        o3 = self.predictor3(x34)

        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear',
                           align_corners=False)  # 根据设定 进行上\下采样  第一个参数input为输入对象，size：输出大小  scale_factor为输出是输入的多少倍 默认false

        o2 = self.predictor2(x234)

        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)

        o1 = self.predictor1(x1234)

        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)

        oe = F.interpolate(edge, scale_factor=4, mode='bilinear', align_corners=False)

        dep1=F.interpolate(dep1, (416,416), mode='bilinear', align_corners=False)
        dep2 = F.interpolate(dep2, (416, 416), mode='bilinear', align_corners=False)
        dep3 = F.interpolate(dep3, (416, 416), mode='bilinear', align_corners=False)
        dep4 = F.interpolate(dep4, (416, 416), mode='bilinear', align_corners=False)

        return  o3, o2, o1, oe,dep4,dep3,dep2,dep1