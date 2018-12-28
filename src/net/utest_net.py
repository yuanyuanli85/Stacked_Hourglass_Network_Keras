
from hg_blocks import create_hourglass_network, bottleneck_block
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

model = create_hourglass_network(num_classes=16, num_stacks=2,
                                 num_channels=256, inres=[256,256],
                                 outres=[64,64], bottleneck=bottleneck_block)

model.summary()