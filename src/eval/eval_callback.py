import keras
import os
import datetime
from time import time

class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath):
        self.foldpath = foldpath


    def get_folder_path(self):
        return self.foldpath

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

