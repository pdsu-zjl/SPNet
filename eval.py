# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from tqdm import tqdm
# pip install pysodmetrics
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
method='BGNet'#44xnbc


d1=['CHAMELEON','CAMO']
d2=['COD10K','NC4K']
d3=['CHAMELEON','CAMO','COD10K']
d4=['NC4K']
d5=['CHAMELEON','CAMO','COD10K','NC4K']

for _data_name in d5:#'NC4K','CHAMELEON','COD10K',
    mask_root = 'E:\COD_attack\ADV\Original_Dataset\{}\GT/'.format(_data_name)
    # pred_root = 'E:\camouflaged object detection\spatial/v2/pred/xiaorong/wubianjiejiandu/{}/'.format(_data_name)
    pred_root =  'E:\camouflaged object detection\spatial/v2\pred2/v5_zishiying2/29\{}/'.format(_data_name)

    mask_name_list = sorted(os.listdir(mask_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):#tqdm（）表示进度条函数  将进度通过进度条根据百分比可视化
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    # results = {
    #     "Smeasure": sm,
    #     "wFmeasure": wfm,
    #     "MAE": mae,
    #     "adpEm": em["adp"],
    #     "meanEm": em["curve"].mean(),
    #     "maxEm": em["curve"].max(),
    #     "adpFm": fm["adp"],
    #     "meanFm": fm["curve"].mean(),
    #     "maxFm": fm["curve"].max(),
    # }

    results = {
        # "name": TTT,
        'dataset':_data_name,
        'methode': _data_name,
        "Smeasure": sm,
        "meanEm": em["curve"].mean(),
        "wFmeasure": wfm,
        "MAE": mae,
    }

    print(results)
    file=open("evalresults.txt", "a")
    file.write(method+' '+_data_name+' '+str(results)+'\n')