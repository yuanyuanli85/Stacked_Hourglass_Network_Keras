

import os
from src.net.hg_blocks import create_hourglass_network, bottleneck_block, bottleneck_mobile
from src.net.hourglass import HourglassNet
import numpy as np
import scipy.misc
from src.data_coco.coco_datagen import CocoDataGen
from src.config.config import AllConfig
from src.data_coco.preprocess import draw_labelmap
from src.eval.heatmap_process import post_process_heatmap
from src.eval.eval_heatmap import cal_heatmap_acc




def main_eval():
    defaultcfg = AllConfig()
    defaultcfg.datacfg.COCO_MIN_BBOX_SIZE = 10000

    xnet = HourglassNet(cfg=defaultcfg)

    xnet.load_model("../../trained_models/hg_s2_b1/net_arch.json",
                    "../../trained_models/hg_s2_b1/weights_epoch42.h5")

    valdata = CocoDataGen(defaultcfg.datacfg, train=False)

    total_good, total_fail = 0, 0
    threshold = 0.5

    print 'val data size', valdata.get_dataset_size()

    count = 0
    batch_size = 8
    for _img, _gthmap, _meta in valdata.generator(batch_size, 8, sigma=1, is_shuffle=False , with_meta=True):

        count += batch_size
        if count % (batch_size*100) == 0:
            print count, 'processed', total_good, total_fail

        if count > valdata.get_dataset_size():
            break

        out = xnet.model.predict(_img)

        good, bad = cal_heatmap_acc(out[-1], _meta, threshold)

        total_good += good
        total_fail += bad

    print total_good, total_fail, threshold, total_good*1.0/(total_good + total_fail)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    main_eval()