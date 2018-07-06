import datetime
import os

from keras.callbacks import CSVLogger
from keras.models import model_from_json

from hg_blocks import create_hourglass_network, bottleneck_block, bottleneck_mobile
from src.config.config import AllConfig
from src.data_coco.coco_datagen import CocoDataGen
from src.eval.eval_callback import EvalCallBack


class HourglassNet(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = cfg.netcfg.CLASS_NUM
        self.num_stacks = cfg.netcfg.STACK_NUM
        self.inres =  (cfg.datacfg.IMAGE_HEIGHT, cfg.datacfg.IMAGE_WIDTH)
        self.outres = (cfg.datacfg.NETWORK_OUT_HEIGHT, cfg.datacfg.NETWORK_OUT_WIDTH)


    def build_model(self, mobile=False, show=False):
        if mobile:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks, self.inres, self.outres,
                                                  self.cfg.traincfg.LEARNING_RATE, bottleneck_mobile)
        else:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks, self.inres, self.outres,
                                                  self.cfg.traincfg.LEARNING_RATE, bottleneck_block)
        # show model summary and layer name
        if show :
            self.model.summary()

    def train(self, batch_size, model_path, epochs):
        train_dataset = CocoDataGen(cfg=self.cfg.datacfg, train=True)
        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma=1, is_shuffle=True,
                                    rot_flag=True, scale_flag=True, flip_flag=True)

        csvlogger = CSVLogger(os.path.join(model_path, "csv_train_"+ str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        xcallbacks = [csvlogger, EvalCallBack(model_path)]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size()//batch_size,
                                 epochs=epochs, callbacks=xcallbacks)

    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):
        pass


    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f :
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)





