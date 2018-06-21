import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../net/")

import os
from keras.utils import plot_model
from keras.utils import plot_model
import numpy as np
import scipy.misc
from mpii_datagen import MPIIDataGen
from heatmap_process import post_process_heatmap
from eval_heatmap import get_predicted_kp_from_htmap, heatmap_accuracy, cal_heatmap_acc
from scipy.io import loadmat
from pckh import run_pckh
from hourglass import HourglassNet

def get_final_pred_kps(valkps, preheatmap, metainfo):

    for i in range(preheatmap.shape[0]):
        prehmap = preheatmap[i, :, :, :]
        meta = metainfo[i]
        sample_index = meta['sample_index']
        kps = get_predicted_kp_from_htmap(prehmap, meta)
        valkps[sample_index, :, :] = kps[:, 0:2] # ignore the visibility

def main_test():
    xnet = HourglassNet(16, 8, (256, 256), (64, 64))

    xnet.load_model("../../trained_models/hg_s8_b1_v1_adam/net_arch.json",
                    "../../trained_models/hg_s8_b1_v1_adam/weights_epoch22.h5")

    valdata = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                inres=(256, 256), outres=(64, 64), is_train=False)

    print 'val data size', valdata.get_dataset_size()

    valkps = np.zeros(shape=(valdata.get_dataset_size(), 16, 2), dtype=np.float)

    count = 0
    batch_size = 8
    for _img, _gthmap, _meta in valdata.generator(batch_size, 8, sigma=2, is_shuffle=False , with_meta=True):

        count += batch_size

        if count > valdata.get_dataset_size():
            break

        out = xnet.model.predict(_img)

        get_final_pred_kps(valkps, out[-1], _meta)


    matfile = os.path.join( "../../trained_models/hg_s8_b1_v1_adam/", 'preds_e22.mat')
    scipy.io.savemat(matfile, mdict={'preds' : valkps})

    run_pckh('hg_s8_b1_epoch22', matfile)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    #main_test()
    matfile = os.path.join( "../../trained_models/hg_s8_b1_v1_adam/", 'preds_e22.mat')
    run_pckh('hg_s8_b1_epoch22', matfile)
