# Stacked_Hourglass_Network_Keras

This is a Keras implementation for stacked hourglass network for single human pose estimation.  The stacked hourglass network was proposed by [Stacked Hourglass Networks for Human Pose Estimation] (https://arxiv.org/abs/1603.06937). The official implementation built on top of torch is released under [pose-hg-train](https://github.com/umich-vl/pose-hg-train), and pytorch version wrote by [berapaw](https://github.com/bearpaw) in repo [pytorch-pose](https://github.com/bearpaw/pytorch-pose). Most of code for image processing and evaluation come from above repos.

## Folder Structure
- `data` : data folder, mpii
- `images` : pictures for demo
- `src` : source code  
`src/data_gen` : data generator, augmentation and processnig code   
`src/eval`: evaluation code, eval callback  
`src/net` : net definition, hourglass network implementation  
`src/tools`: tool to draw accuracy curve and convert keras model to tf graph.  
`top`: top level entry to train/eval/demo network
- `trained_models` : folder to restore trained models.

## Demo

## Train
#### MPII Data Preparation
- Download MPII Dataset and put its images under `data/mpii/images`
- The json `mpii_annotations.json` contains all of images' annotations including train and validation.

#### Train network
- Train from scratch, use `python train.py --help` to check all the valid arguments.   
```
python train.py --gpuID 0 --epochs 100 --batch_size 2 --num_stack 2 --model_path ../../trained_models/hg_s2_b1_m
```
- Arguments:  
`gpuID` gpu id, `epochs` number of epoch to train, `batch_size` batch size of samples to train,  
`num_stack` number of hourglass stack, `model_path` path to store trained model snapshot  
- Note:  
When `mobile` set as True, `SeparableConv2D()` is used instead of standard convolution, which is much smaller and faster.


- Continue training from previous checkpoint  
```
python train.py --gpuID 0 --epochs 100 --batch_size 2 --num_stack 2 --model_path ../../trained_models/hg_s2_b1_m  --resume True --resume_model_json ../../trained_models/hg_s2_b1_m/net_arch.json --resume_model ../../trained_models/hg_s2_b1_m/weights_epoch15.h5 --init_epoch 16
```

## Eval
