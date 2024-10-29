import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.SPNet import Net
from utils.tdataloader import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size') #测试规模
parser.add_argument('--pth_path', type=str, default='./checkpoint/SPNet-best.pth')#图片路径变量

# for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:



for _data_name in ['CAMO','COD10K']:
        # data_path = 'E:\COD_attack\ADV\Original_Dataset\{}/Imgs/'.format(_data_name)
        data_path='./{}'.format(_data_name)
        depth_path='E:\COD_attack\ADV\Original_Dataset\{}\Depth/'.format(_data_name)
        save_path = './{}'.format(_data_name)
        opt = parser.parse_args() #创建对象
        model = Net()
        model.load_state_dict(torch.load(opt.pth_path)) #加载模型参数
        model.cuda()
        model.eval()

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path+'PRE/', exist_ok=True)
        image_root = data_path
        gt_root = 'E:\COD_attack\ADV\Original_Dataset/{}/GT/'.format(_data_name)
        test_loader = test_dataset(image_root, gt_root,depth_path, opt.testsize)
        print(test_loader.size)
        for i in range(test_loader.size):
            image, gt, depth,name = test_loader.load_data() #测试数据集+图片预测结果
            gt = np.asarray(gt, np.float32)  #将结构数据转化为ndarray
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth=depth.cuda()
            _, _, res,e,t1,t2,t3,t4= model(image,depth)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)  #upsample  可以采样直接塑形   设置自己要的size不用顾虑那么多是吧
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imwrite(save_path+name, (res*255).astype(np.uint8))
            e = F.interpolate(e, size=gt.shape, mode='bilinear', align_corners=True)
            e = e.data.cpu().numpy().squeeze()   #squeeze  除去size为1的维度
            e = (e - e.min()) / (e.max() - e.min() + 1e-8)
            imageio.imwrite(save_path+'PRE/'+name, (e*255).astype(np.uint8))