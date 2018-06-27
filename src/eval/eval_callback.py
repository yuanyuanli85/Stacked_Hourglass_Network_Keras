import keras
import os
import datetime
from time import time
from mpii_datagen import MPIIDataGen
from eval_heatmap import cal_heatmap_acc

class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath):
        self.foldpath = foldpath


    def get_folder_path(self):
        return self.foldpath

    def run_eval(self, epoch, mask_flag):
        valdata = MPIIDataGen("../../data/mpii/mpii_annotations.json",
                              "../../data/mpii/images",
                              inres=(256, 256), outres=(64, 64), is_train=False)

        total_suc, total_fail = 0, 0
        threshold = 0.5

        count = 0
        batch_size = 8

        num_stack = len(self.model.output_layers)
        for _input, _gthmap, _meta in valdata.generator(batch_size, num_stack, sigma=2, is_shuffle=False,
                                                      with_meta=True, mask_flag=mask_flag):

            count += batch_size
            if count > valdata.get_dataset_size():
                break

            out = self.model.predict(_input)

            suc, bad = cal_heatmap_acc(out[-1], _meta, threshold)

            total_suc += suc
            total_fail += bad

        acc = total_suc*1.0 / (total_fail + total_suc)

        print 'Eval Accuray ', acc, '@ Epoch ', epoch

        with open(os.path.join(self.get_folder_path(), 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + '\n')

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

        self.run_eval(epoch, self.check_mask_in_input())


    def check_mask_in_input(self):
        input_num = self.model.input_layers
        if input_num > 1:
            return True
        return False