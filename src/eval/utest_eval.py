import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../net/")

import os
from hg_blocks import create_hourglass_network
from keras.utils import plot_model
from keras.utils import plot_model
from hourglass import HourglassNet
import numpy as np
import scipy.misc
from mpii_datagen import MPIIDataGen
from heatmap_process import post_process_heatmap
from eval_heatmap import get_predicted_kp_from_htmap, heatmap_accuracy, cal_heatmap_acc

def main():
    model = create_hourglass_network(16, 2, (256, 256), (64, 64))
    model.summary()
    print len(model.output_layers)
    plot_model(model, 'hg_s2.png', show_shapes=True)
    for layer in model.output_layers:
        print layer.output_shape

def view_predict_hmap(predout, show_raw=False):
    from data_process import draw_labelmap
    gtmap = predout[-1]
    gtmap = gtmap[0,:,:,:]

    if show_raw:
        mimage = gtmap
    else:
        mimage = np.zeros(shape=gtmap.shape, dtype=np.float)
        kplst = post_process_heatmap(gtmap)
        for i, kpoint in enumerate(kplst):
            mimage[:,:,i] = draw_labelmap(mimage[:,:,i], kpoint, sigma=2)

    sumimage = np.zeros(shape=(64, 64))
    for i in range(gtmap.shape[-1]):
        sumimage += mimage[:,:,i]
    scipy.misc.imshow(sumimage)



def main_test():
    xnet = HourglassNet(16, 8, (256, 256), (64, 64))

    xnet.load_model("../../trained_models/hg_s8_b1/net_arch.json", "../../trained_models/hg_s8_b1/weights_epoch60.h5")

    valdata = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                inres=(256, 256), outres=(64, 64), is_train=False)

    total_good, total_fail = 0, 0
    threshold = 0.5

    print 'val data size', valdata.get_dataset_size()

    count = 0
    batch_size = 8
    for _img, _gthmap, _meta in valdata.generator(batch_size, 8, sigma=2, is_shuffle=False , with_meta=True):

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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main_test()