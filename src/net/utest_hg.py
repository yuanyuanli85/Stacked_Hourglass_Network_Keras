import os
from hg_blocks import create_hourglass_network
from keras.utils import plot_model
from keras.utils import plot_model
from hourglass import HourglassNet
import numpy as np
import scipy.misc
from mpii_datagen import MPIIDataGen
from scipy.ndimage import gaussian_filter, maximum_filter


def main():
    model = create_hourglass_network(16, 2, (256, 256), (64, 64))
    model.summary()
    print len(model.output_layers)
    plot_model(model, 'hg_s2.png', show_shapes=True)
    for layer in model.output_layers:
        print layer.output_shape

def post_process_heatmap(heatMap, kpConfidenceTh=0.2):
    kplst = list()
    for i in range(heatMap.shape[-1]):
        # ignore last channel, background channel
        _map = heatMap[:, :, i]
        _map = gaussian_filter(_map, sigma=0.5)
        _nmsPeaks = non_max_supression(_map, windowSize=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        kplst.append((int(x[0]), int(y[0]), 1))
    return kplst

def non_max_supression(plain, windowSize=3, threshold=1e-6):
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain* (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))

def view_predict_hmap(predout):
    from data_process import draw_labelmap
    gtmap = predout[0]
    gtmap = gtmap[0,:,:,:]

    mimage = np.zeros(shape=gtmap.shape, dtype=np.float)

    kplst = post_process_heatmap(gtmap)
    for i, kpoint in enumerate(kplst):
        mimage[:,:,i] = draw_labelmap(mimage[:,:,i], kpoint, sigma=2)

    sumimage = np.zeros(shape=(64, 64))
    for i in range(gtmap.shape[-1]):
        sumimage += mimage[:,:,i]
    scipy.misc.imshow(sumimage)



def main_test():
    xnet = HourglassNet(16, 4, (256, 256), (64, 64))
    xnet.load_model("../../trained_models/weights_10_63.97.hdf5")

    valdata = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                inres=(256, 256), outres=(64, 64), is_train=False)

    for _img, _gthmap in valdata.generator(1, 4, sigma=2, is_shuffle=False):
        out = xnet.model.predict(_img)

        scipy.misc.imshow(_img[0,:,:,:])
        #view_predict_hmap(_gthmap)
        view_predict_hmap(out)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    main_test()