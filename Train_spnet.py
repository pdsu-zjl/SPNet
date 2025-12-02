import torch
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
# from net.spnet9 import Net
from mokuai.spnet12 import Net
# from mokuai.MSCANet_albation.RGB import Net
from dual_backbone.dual import Net

from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np

file = open("log/BGNet.txt", "a")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True


def structure_loss(pred, mask):  #结构损失  mask掩码  pred---》预测图   mask---》gt图
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  #1+5*|avg_pool(mask) -mask|  ||绝对值
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')  #交叉熵损失
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)) #沿着第三第四维度加和  得到

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))  #乘
    union = ((pred + mask) * weit).sum(dim=(2, 3))  #加
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):  #dic损失  预测结果和目标值
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)  #创建和 target形状一致的张量
    predict = predict.contiguous().view(predict.shape[0], -1)  #view（） 相当于 reshape   -1表示不确定的地方  计算后确定
    target = target.contiguous().view(target.shape[0], -1)       #view变形为二维数组
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth  #pow（x） 返回目标的x次方
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch):
    model.train()

    loss_record3, loss_record2, loss_record1, loss_recorde = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()  # AvgMeter()  计算一个epoch的平均损失
    for i, pack in enumerate(train_loader, start=1):  #枚举 从1开始
        optimizer.zero_grad()  #梯度清零
        # ---- data prepare ----
        images, gts, edges,depths = pack

        images = Variable(images).cuda()  #Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现(tensor变成variable之后才能进行反向传播求梯度?用变量.backward()进行反向传播之后,var.grad中保存了var的梯度)
        gts = Variable(gts).cuda()        #封装作用  图片--人工图（作为掩码）---边界图
        edges = Variable(edges).cuda()
        depths = Variable(depths).cuda()

        # ---- forward ----
        lateral_map_3, lateral_map_2, lateral_map_1, edge_map,dep4,dep3,dep2,dep1 = model(images,depths)
        # o3             o2             o1             oe--》edge   最后使用o1 和 oe进行测试
        # ---- loss function ----
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss1 = structure_loss(lateral_map_1, gts)  #损失   损失函数都是别人的  去cod方向里面的论文调用即可
        losse_d = dice_loss(edge_map, edges)

        lossd1 = structure_loss(dep1, gts)
        lossd2 = structure_loss(dep2, gts)
        lossd3 = structure_loss(dep3, gts)
        lossd4 = structure_loss(dep4, gts)

        loss = loss3 + loss2 + loss1*3 + losse_d+lossd1+lossd2+lossd3+lossd4
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)   #优化
        optimizer.step()
        # ---- recording loss ----
        loss_record3.update(loss3.data, opt.batchsize)   #参数更新
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record1.update(loss1.data, opt.batchsize)
        loss_recorde.update(losse_d.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step: #为60的倍数  或者达到总数 打印一个迭代次数的内容
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '          
                  '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,   #一个{}对应一项内容  03d表示三个字符长度
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))
            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]\n'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))

    save_path = 'checkpoints/{}/'.format(opt.train_save)  #保存路径
    os.makedirs(save_path, exist_ok=True)              # 保存目录
    if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'BGNet-%d.pth' % epoch)  #保存一个序列化（serialized）的目标到磁盘  1：保存对象  2：类文件对象（路径）  保存模型参数
        print('[Saving Snapshot:]', save_path + 'BGNet-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'BGNet-%d.pth' % epoch + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()  #parser  argparse的对象
    parser.add_argument('--epoch', type=int,              #在parser的对象内部添加属性  （1）name ：名字（2）action：动作，默认值是store（3）default：默认值（4）type:参数的类型（5）help：参数的帮助信息
                        default=50, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=1, help='training batch size')#16
    parser.add_argument('--trainsize', type=int,
                        default=672, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='BGNet')
    opt = parser.parse_args()  #创建一个新对象 opt

    # ---- build models ----
    net=Net()
    model =net.cuda()
       #调用gpu + 模型参数
    params = model.parameters()  #参数获取

    optimizer = torch.optim.Adam(params, opt.lr) #adam训练器  ？？？理解的不够  优化算法
                                # 优化的参数   学习率
    # path='/root'
    # path='E:\camouflaged object detection/TrainDataset/'
    path='E:\COD_attack\ADV\Original_Dataset\COD10K'
    image_root = path+'/Imgs/'.format(opt.train_path) #数据的路径
    gt_root = path+'/GT/'.format(opt.train_path)
    edge_root = path+'/Edge/'.format(opt.train_path)
    depth_root=path+'/Depth/'.format(opt.train_path)
                                                                #批量 16                      训练大小  416
    train_loader = get_loader(image_root, gt_root, edge_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader) #长度

    print("Start Training")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)  #对学习率进行微调 然后返回
        train(train_loader, model, optimizer, epoch)

    file.close()
