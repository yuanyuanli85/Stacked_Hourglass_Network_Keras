import os
import numpy as np
from random import shuffle
import scipy.misc
import json
import data_process


class MPIIDataGen(object):

    def __init__(self, jsonfile, imgpath, inres, outres, is_train):
        self.jsonfile = jsonfile
        self.imgpath  = imgpath
        self.inres    = inres
        self.outres   = outres
        self.is_train = is_train
        self.nparts   = 16
        self.anno     = self._load_image_annotation()

    def _load_image_annotation(self):
        # load train or val annotation
        with open(self.jsonfile) as anno_file:
            anno = json.load(anno_file)

        val_anno, train_anno = [], []
        for idx, val in enumerate(anno):
            if val['isValidation'] == True:
                val_anno.append(anno[idx])
            else:
                train_anno.append(anno[idx])

        if self.is_train:
            return train_anno
        else:
            return val_anno

    def get_dataset_size(self):
        return len(self.anno)

    def get_annotations(self):
        return self.anno

    def generator(self, batch_size, num_hgstack, sigma=1, is_shuffle=False):
        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''
        train_input = np.zeros(shape=(batch_size, self.inres[0], self.inres[1], 3), dtype=np.float)
        gt_heatmap  = np.zeros(shape=(batch_size, self.outres[0], self.outres[1], self.nparts), dtype=np.float)

        while True:
            if is_shuffle:
                shuffle(self.anno)

            for i, kpanno in enumerate(self.anno):
                #with Timer():
                _imageaug, _gthtmap = self.process_image(kpanno, sigma)
                _index = i%batch_size

                train_input[_index , :, :, : ] = _imageaug
                gt_heatmap[_index, :, :, :] = _gthtmap

                if i != 0 and i%batch_size == 0:
                    out_hmaps = []
                    for m in range(num_hgstack):
                        out_hmaps.append(gt_heatmap)
                    yield train_input, out_hmaps


    def process_image(self, kpanno, sigma):
        imagefile = kpanno['img_paths']
        image = scipy.misc.imread(os.path.join(self.imgpath, imagefile))

        # get center
        center = np.array(kpanno['objpos'])
        scale =  kpanno['scale_provided']
        rot = 0

        # crop image
        cropimg = data_process.crop(image, center, scale, self.inres, rot)
        cropimg = data_process.normalize(cropimg)

        # transform keypoints
        transformedKps = data_process.transform_kp(np.array(kpanno['joint_self']), center, scale, self.outres, rot)
        gtmap = data_process.generate_gtmap(transformedKps, sigma, self.outres)

        return cropimg, gtmap





