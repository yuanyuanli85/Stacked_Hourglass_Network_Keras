from random import shuffle, choice
import numpy as np
from coco_anno import get_coco_kp_annotation
from preprocess import crop_image, normalize_image, generate_gtmap, rotate_image, flip_image


class CocoDataGen(object):

    def __init__(self, cfg, train):
        self.cfg = cfg
        self.is_train = train
        self.load_annotations()

    def load_annotations(self):
        if self.is_train:
            self.anno = get_coco_kp_annotation(self.cfg.COCO_TRAIN_ANNO_JSON,
                                               self.cfg.COCO_TRAIN_IMAGE_PATH,
                                               self.cfg)
        else:
            self.anno = get_coco_kp_annotation(self.cfg.COCO_VAL_ANNO_JSON,
                                               self.cfg.COCO_VAL_IMAGE_PATH,
                                               self.cfg)

    def get_dataset_size(self):
        return len(self.anno)


    def generator(self, batch_size, num_hgstack, sigma=1, with_meta=False, is_shuffle=False,
                  rot_flag=False, scale_flag=False, flip_flag=False):
        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''
        train_input = np.zeros(shape=(batch_size, self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH, 3), dtype=np.float)
        gt_heatmap  = np.zeros(shape=(batch_size, self.cfg.NETWORK_OUT_HEIGHT, self.cfg.NETWORK_OUT_WIDTH, self.cfg.PART_NUM), dtype=np.float)
        meta_info   = list()

        if not self.is_train:
            assert (is_shuffle == False), 'shuffle must be off in val model'
            assert (rot_flag == False),  'rot_flag must be off in val model'

        while True:
            if is_shuffle:
                shuffle(self.anno)

            for i, kpanno in enumerate(self.anno):
                _imageaug, _gthtmap, _meta = self.process_image(i, kpanno, sigma, rot_flag, scale_flag, flip_flag)
                _index = i%batch_size

                train_input[_index, :, :, :] = _imageaug
                gt_heatmap[_index, :, :, :] = _gthtmap
                meta_info.append(_meta)

                if i%batch_size == (batch_size -1):
                    out_hmaps = []
                    for m in range(num_hgstack):
                        out_hmaps.append(gt_heatmap)

                    if with_meta:
                        yield train_input, out_hmaps, meta_info
                        meta_info = []
                    else:
                        yield train_input, out_hmaps


    def process_image(self, sample_index, kpanno, sigma, rot_flag, scale_flag, flip_flag):
        # crop image
        cvmat, keypoints = crop_image(kpanno, cfg=self.cfg)
        cvmat = normalize_image(cvmat)

        #rotate image
        if rot_flag and choice([0, 1]):
            angle = np.random.randint(-1*30, 30)
            cvmat, keypoints = rotate_image(cvmat, keypoints, angle, self.cfg)

        #flip image
        if flip_flag and choice([0, 1]):
            cvmat, keypoints = flip_image(cvmat, keypoints, self.cfg.COCO_SYMMETRY_PARIS)

        # downsample to 1/4 for ground truth heatmap
        gtkpoints = np.copy(keypoints)
        gtkpoints[:,0] = gtkpoints[:,0]/4.0
        gtkpoints[:,1] = gtkpoints[:,1]/4.0
        gtkpoints = gtkpoints.astype(np.int)

        # generate heatmap
        gtmap = generate_gtmap(gtkpoints, sigma,
                                 (self.cfg.NETWORK_OUT_HEIGHT, self.cfg.NETWORK_OUT_WIDTH))
        # meta info
        metainfo =  { 'sample_index':sample_index, 'pts': keypoints, 'tpts':gtkpoints, 'name' : kpanno['filename']}

        return cvmat, gtmap, metainfo
