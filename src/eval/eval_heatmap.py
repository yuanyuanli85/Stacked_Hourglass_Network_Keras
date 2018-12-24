from heatmap_process import post_process_heatmap
import data_process
import numpy as np
import copy


def get_predicted_kp_from_htmap(heatmap, meta, outres):
    # nms to get location
    kplst = post_process_heatmap(heatmap)
    kps = np.array(kplst)

    # use meta information to transform back to original image
    mkps = copy.copy(kps)
    for i in range(kps.shape[0]):
        mkps[i, 0:2] = data_process.transform(kps[i], meta['center'], meta['scale'], res=outres, invert=1, rot=0)

    return mkps


def cal_kp_distance(pre_kp, gt_kp, norm, threshold):
    if gt_kp[0] > 1 and gt_kp[1] > 1:
        dif = np.linalg.norm(gt_kp[0:2] - pre_kp[0:2]) / norm
        if dif < threshold:
            # good prediction
            return 1
        else:  # failed
            return 0
    else:
        return -1


def heatmap_accuracy(predhmap, meta, norm, threshold):
    pred_kps = post_process_heatmap(predhmap)
    pred_kps = np.array(pred_kps)

    gt_kps = meta['tpts']

    good_pred_count = 0
    failed_pred_count = 0
    for i in range(gt_kps.shape[0]):
        dis = cal_kp_distance(pred_kps[i, :], gt_kps[i, :], norm, threshold)
        if dis == 0:
            failed_pred_count += 1
        elif dis == 1:
            good_pred_count += 1

    return good_pred_count, failed_pred_count


def cal_heatmap_acc(prehmap, metainfo, threshold):
    sum_good, sum_fail = 0, 0
    for i in range(prehmap.shape[0]):
        _prehmap = prehmap[i, :, :, :]
        good, bad = heatmap_accuracy(_prehmap, metainfo[i], norm=6.4, threshold=threshold)

        sum_good += good
        sum_fail += bad

    return sum_good, sum_fail
