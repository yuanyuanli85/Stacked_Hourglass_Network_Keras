from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
import keras.backend as K

def create_hourglass_network(num_classes, num_stacks, inres, outres, bottleneck):

    input = Input(shape=(inres[0], inres[1], 3))

    front_features = create_front_module(input, bottleneck)

    head_next_stage = front_features

    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, bottleneck, i)
        outputs.append(head_to_loss)
        '''
        if i == num_stacks - 1:
            # last stage, append last head
            last_head = Conv2D(num_classes, kernel_size=(1,1), activation='linear', padding='same') (head_next_stage)
            outputs.append(last_head)
        else:
            outputs.append(head_to_loss)
        '''
    model = Model(inputs=input, outputs=outputs)
    rms = RMSprop(lr=5e-4)
    model.compile(optimizer=rms, loss=mean_squared_error, metrics=["accuracy"])

    return model

def hourglass_module(bottom, num_classes, bottleneck, hgid):
    # create left features , f1, f2, f4, and f8
    left_features = create_left_half_blocks(bottom, bottleneck, hgid)

    # create right features, connect with left features
    rf1 = create_right_half_blocks(left_features, bottleneck, hgid)

    # add 1x1 conv with two heads, head_next_stage is sent to next stage
    # head_parts is used for intermediate supervision
    head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hgid)

    return head_next_stage, head_parts

def bottleneck_block(bottom, num_out_channels, block_name):
    # skip layer
    if K.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = Conv2D(num_out_channels, kernel_size=(1,1), activation='relu', padding='same', name=block_name+'skip') (bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = Conv2D(num_out_channels/2, kernel_size=(1,1), activation='relu', padding='same', name=block_name+'_conv_1x1_x1') (bottom)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels/2, kernel_size=(3,3), activation='relu', padding='same', name=block_name+'_conv_3x3_x2') (_x)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels, kernel_size=(1,1), activation='relu', padding='same', name=block_name+'_conv_1x1_x3') (_x)
    _x = BatchNormalization()(_x)
    _x = Add(name=block_name+'_residual')([_skip, _x])

    return _x


def bottleneck_mobile(bottom, num_out_channels, block_name):
    # skip layer
    if K.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = SeparableConv2D(num_out_channels, kernel_size=(1,1), activation='relu', padding='same', name=block_name+'skip') (bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = SeparableConv2D(num_out_channels/2, kernel_size=(1,1), activation='relu', padding='same', name=block_name+'_conv_1x1_x1') (bottom)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_channels/2, kernel_size=(3,3), activation='relu', padding='same', name=block_name+'_conv_3x3_x2') (_x)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_channels, kernel_size=(1,1), activation='relu', padding='same', name=block_name+'_conv_1x1_x3') (_x)
    _x = BatchNormalization()(_x)
    _x = Add(name=block_name+'_residual')([_skip, _x])

    return _x

def create_front_module(input, bottleneck):
    # front module, input to 1/4 resolution
    # 1 7x7 conv + maxpooling
    # 3 residual block

    _x = Conv2D(64, kernel_size=(7,7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1_x1') (input)
    _x = BatchNormalization()(_x)

    _x = bottleneck(_x, 128, 'front_residual_x1')
    _x = MaxPool2D(pool_size=(2,2), strides=(2,2))(_x)

    _x = bottleneck(_x, 128, 'front_residual_x2')
    _x = bottleneck(_x, 256, 'front_residual_x3')

    return _x

def create_left_half_blocks(bottom, bottleneck, hglayer):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    hgname = 'hg'+str(hglayer)
    num_channles = 256

    f1 = bottleneck(bottom, num_channles, hgname+'_l1')
    _x = MaxPool2D(pool_size=(2,2), strides=(2,2))(f1)

    f2 = bottleneck(_x, num_channles, hgname+'_l2')
    _x = MaxPool2D(pool_size=(2,2), strides=(2,2))(f2)

    f4 = bottleneck(_x, num_channles, hgname + '_l4')
    _x = MaxPool2D(pool_size=(2,2), strides=(2,2))(f4)

    f8 = bottleneck(_x, num_channles, hgname + '_l8')

    return (f1, f2, f4 , f8)

def connect_left_to_right(left, right, bottleneck, name):
    '''
    :param left: connect left feature to right feature
    :param name: layer name
    :return:
    '''
    # left -> 1 bottlenect
    # right -> upsampling
    # Add   -> left + right

    _xleft  = bottleneck(left, 256, name+'_connect')
    _xright = UpSampling2D()(right)
    add = Add()([_xleft, _xright])
    out = bottleneck(add, 256, name+'_connect_conv')
    return out

def bottom_layer(lf8, bottleneck, hgid):
    # blocks in lowest resolution
    # 3 bottlenect blocks + Add

    lf8_connect = bottleneck(lf8, 256, str(hgid)+"_lf8")

    _x = bottleneck(lf8, 256, str(hgid)+"_lf8_x1")
    _x = bottleneck(_x,  256, str(hgid)+"_lf8_x2")
    _x = bottleneck(_x,  256, str(hgid)+"_lf8_x3")

    rf8 = Add()([_x, lf8_connect])

    return rf8

def create_right_half_blocks(leftfeatures, bottleneck, hglayer):
    lf1, lf2, lf4, lf8 =  leftfeatures

    rf8 = bottom_layer(lf8, bottleneck, hglayer)

    rf4 = connect_left_to_right(lf4, rf8, bottleneck, 'hg'+str(hglayer)+'_rf4')

    rf2 = connect_left_to_right(lf2, rf4, bottleneck, 'hg'+str(hglayer)+'_rf2')

    rf1 = connect_left_to_right(lf1, rf2, bottleneck, 'hg'+str(hglayer)+'_rf1')

    return rf1

def create_heads(prelayerfeatures, rf1, num_classes, hgid):
    # two head, one head to next stage, one head to intermediate features
    head =  Conv2D(256, kernel_size=(1,1), activation='relu', padding='same', name=str(hgid)+'_conv_1x1_x1') (rf1)
    head = BatchNormalization()(head)

    # for head as intermediate supervision, use 'linear' as activation.
    head_parts = Conv2D(num_classes, kernel_size=(1,1), activation='linear', padding='same', name=str(hgid)+'_conv_1x1_parts') (head)

    # use linear activation
    head = Conv2D(256, kernel_size=(1,1), activation='linear', padding='same', name=str(hgid)+'_conv_1x1_x2') (head)
    head_m = Conv2D(256, kernel_size=(1,1), activation='linear', padding='same', name=str(hgid)+'_conv_1x1_x3') (head_parts)

    head_next_stage = Add()([head, head_m, prelayerfeatures])
    return head_next_stage, head_parts


def euclidean_loss(x, y):
        return K.sqrt(K.sum(K.square(x - y)))
