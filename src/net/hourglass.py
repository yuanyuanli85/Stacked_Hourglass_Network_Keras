import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../eval/")


import os
from hg_blocks import create_hourglass_network, euclidean_loss
from mpii_datagen import MPIIDataGen
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model, model_from_json
import datetime
import scipy.misc
from data_process import normalize
import numpy as np
from eval_callback import EvalCallBack

class HourglassNet(object):

    def __init__(self, num_classes, num_stacks, inres, outres):
        self.num_classes = num_classes
        self.num_stacks = num_stacks
        self.inres = inres
        self.outres = outres


    def build_model(self, show=False):
        self.model = create_hourglass_network(self.num_classes, self.num_stacks, self.inres, self.outres)
        # show model summary and layer name
        if show :
            self.model.summary()

    def train(self, batch_size, model_path, epochs):
        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                      inres=self.inres,  outres=self.outres, is_train=True)
        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma=2, is_shuffle=True)

        csvlogger = CSVLogger(os.path.join(model_path, "csv_train_"+ str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))
        modelfile = os.path.join(model_path, 'weights_{epoch:02d}_{loss:.2f}.hdf5')

        checkpoint =  EvalCallBack(model_path)

        xcallbacks = [csvlogger, checkpoint]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size()//batch_size,
                                 epochs=epochs, callbacks=xcallbacks)

    def resume_train(self):
        pass


    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f :
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)

    '''
    def load_model(self, modelfile):
            self.model = load_model(modelfile, custom_objects={'euclidean_loss': euclidean_loss})
    '''

    def predict(self, imgfile):
        imgdata = scipy.misc.imread(imgfile)
        imgdata = normalize(imgdata)
        imgdata = scipy.misc.imresize(imgdata, self.inres)
        input = imgdata[np.newaxis,:,:,:]

        out = self.model.predict(input)
        return out

