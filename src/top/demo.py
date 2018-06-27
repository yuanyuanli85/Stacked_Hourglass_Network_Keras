import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../net/")
sys.path.insert(0, "../eval/")

import os
import numpy as np
import scipy.misc
from heatmap_process import post_process_heatmap
from hourglass import HourglassNet
import argparse
from pckh import run_pckh
from mpii_datagen import MPIIDataGen
import cv2


def render_joints(cvmat, joints):
    for _joint in joints:
        _x, _y , _v = _joint
        if _v == 1:
            cv2.circle(cvmat, center=(int(_x), int(_y)), color=(1.0, 1.0, 0), radius=7, thickness=2)

    return cvmat

def main_inference(model_json, model_weights, num_stack, num_class, imgfile):
    xnet = HourglassNet(num_class, num_stack, (256, 256), (64, 64))
    xnet.load_model(model_json, model_weights)

    out, scale = xnet.inference_file(imgfile)

    print out.shape, scale

    kps = post_process_heatmap(out[0,:,:,:])

    ignore_kps = ['plevis', 'thorax', 'head_top']
    kp_keys = MPIIDataGen.get_kp_keys()
    mkps = list()
    for i, _kp in enumerate(kps):
        if kp_keys[i] in ignore_kps:
            _v = -1
        else:
            _v = 1
        mkps.append((_kp[0]*scale[1]*4, _kp[1]*scale[0]*4, _v))

    display_joints(imgfile, mkps)


def main_video(model_json, model_weights, num_stack, num_class, videofile):

    xnet = HourglassNet(num_class, num_stack, (256, 256), (64, 64))
    xnet.load_model(model_json, model_weights)

    cap = cv2.VideoCapture(videofile)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            rgb = frame[:,:,::-1] # bgr -> rgb
            out, scale = xnet.inference_rgb(rgb, frame.shape)

            kps = post_process_heatmap(out[0, :, :, :])

            ignore_kps = ['plevis', 'thorax', 'head_top']
            kp_keys = MPIIDataGen.get_kp_keys()
            mkps = list()
            for i, _kp in enumerate(kps):
                if kp_keys[i] in ignore_kps:
                    _v = -1
                else:
                    _v = 1
                mkps.append((_kp[0] * scale[1] * 4, _kp[1] * scale[0] * 4, _v))

            framejoints = render_joints(frame, mkps)

            cv2.imshow('frame', framejoints)
            cv2.waitKey(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--model_json",  help='path to store trained model')
    parser.add_argument("--model_weights",  help='path to store trained model')
    parser.add_argument("--num_stack",  type=int, help='num of stack')
    parser.add_argument("--input_image",  help='input image file')
    parser.add_argument("--input_video", default='', help='input video file')


    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    if args.input_image:
       main_inference(model_json=args.model_json, model_weights=args.model_weights, num_stack=args.num_stack,
                   num_class=16, imgfile = args.input_image)
    elif args.input_video:
        main_video(model_json=args.model_json, model_weights=args.model_weights, num_stack=args.num_stack,
                   num_class=16, videofile=args.input_video)