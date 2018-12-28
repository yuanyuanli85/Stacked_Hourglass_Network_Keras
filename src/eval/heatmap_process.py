from scipy.ndimage import gaussian_filter, maximum_filter
import numpy as np


def post_process_heatmap(heatMap, kpConfidenceTh=0.2):
    kplst = list()
    for i in range(heatMap.shape[-1]):
        _map = heatMap[:, :, i]
        _map = gaussian_filter(_map, sigma=0.5)
        _nmsPeaks = non_max_supression(_map, windowSize=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        if len(x) > 0 and len(y) > 0:
            kplst.append((x[0], y[0], _nmsPeaks[y[0], x[0]]))
        else:
            kplst.append((0, 0, 0))

    kps = refine_preds_keypoints(kplst, heatMap)
    return kps


def non_max_supression(plain, windowSize=3, threshold=1e-6):
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))


def refine_preds_keypoints(kplst, heatMap):
    kp  = np.array(kplst)

    for i in range(kp.shape[0]):
        hm = heatMap[:,:,i]
        px = int(kp[i][0])
        py = int(kp[i][1])
        if px > 1 and px < heatMap.shape[1] and py > 1 and py < heatMap.shape[0]:
            diff = [hm[py - 1][px] - hm[py - 1][px - 2],
                    hm[py][px - 1] - hm[py - 2][px - 1]]
            diff = np.array(diff)
            kp[i,:-1] +=  np.sign(diff)*0.25
    kp += 0.5
    return kp