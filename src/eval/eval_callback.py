import keras
import os
import datetime
from time import time
from mpii_datagen import MPIIDataGen
from eval_heatmap import cal_heatmap_acc, get_final_pred_kps
import numpy as np
import scipy.io
from pckh import run_pckh

class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath, inres, outres):
        self.foldpath = foldpath
        self.inres = inres
        self.outres = outres

    def get_folder_path(self):
        return self.foldpath

    def run_eval(self, epoch):
        valdata = MPIIDataGen("../../data/mpii/mpii_annotations.json",
                              "../../data/mpii/images",
                              inres=self.inres, outres=self.outres, is_train=False)

        total_suc, total_fail = 0, 0
        threshold = 0.5

        count = 0
        batch_size = 8

        valkps = np.zeros(shape=(valdata.get_dataset_size(), 16, 2), dtype=np.float)

        for _img, _gthmap, _meta in valdata.generator(batch_size, 8, sigma=1, is_shuffle=False, with_meta=True):

            count += batch_size
            if count > valdata.get_dataset_size():
                break

            out = self.model.predict(_img)

            suc, bad = cal_heatmap_acc(out[-1], _meta, threshold)

            get_final_pred_kps(valkps, out[-1], _meta, self.outres)

            total_suc += suc
            total_fail += bad

        acc = total_suc * 1.0 / (total_fail + total_suc)
        print 'Eval Accuray ', acc, '@ Epoch ', epoch
        with open(os.path.join(self.get_folder_path(), 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + '\n')

        # save to matfile and do pckh eval
        matfile = os.path.join(self.foldpath, "preds_epoch{}.mat".format(epoch))
        scipy.io.savemat(matfile, mdict={'preds': valkps})
        evalstr = run_pckh('Epoch:'+str(epoch), matfile)
        with open(os.path.join(self.get_folder_path(), 'val.txt'), 'a+') as xfile:
            xfile.write(evalstr+ '\n')

    def on_epoch_end(self, epoch, logs=None):
        # This is a walkaround to sovle model.save() issue
        # in which large network can't be saved due to size.

        # save model to json
        if epoch == 0:
            jsonfile = os.path.join(self.foldpath, "net_arch.json")
            with open(jsonfile, 'w') as f:
                f.write(self.model.to_json())

        # save weights
        modelName = os.path.join(self.foldpath, "weights_epoch" + str(epoch) + ".h5")
        self.model.save_weights(modelName)

        print "Saving model to ", modelName

        self.run_eval(epoch)
