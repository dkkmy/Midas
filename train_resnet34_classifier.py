from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as np_utils
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from helper import get_spectograms
import numpy as np
import os

def conv_block(inputs, 
        neuron_num, 
        kernel_size,  
        use_bias, 
        padding= 'same',
        strides= (1, 1),
        with_conv_short_cut = False):
    conv1 = Conv2D(
        neuron_num,
        kernel_size = kernel_size,
        activation= 'relu',
        strides= strides,
        use_bias= use_bias,
        padding= padding
    )(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)

    conv2 = Conv2D(
        neuron_num,
        kernel_size= kernel_size,
        activation= 'relu',
        use_bias= use_bias,
        padding= padding)(conv1)
    conv2 = BatchNormalization(axis = 1)(conv2)

    if with_conv_short_cut:
        inputs = Conv2D(
            neuron_num, 
            kernel_size= kernel_size,
            strides= strides,
            use_bias= use_bias,
            padding= padding
            )(inputs)
        return add([inputs, conv2])

    else:
        return add([inputs, conv2])

inputs = Input(shape= [224, 216, 1])
x = ZeroPadding2D((3, 3))(inputs)

# Define the converlutional block 1
x = Conv2D(64, kernel_size= (7, 7), strides= (2, 2), padding= 'valid')(x)
x = BatchNormalization(axis= 1)(x)
x = MaxPooling2D(pool_size= (3, 3), strides= (2, 2), padding= 'same')(x)

# Define the converlutional block 2
x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)

# Define the converlutional block 3
x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)

# Define the converlutional block 4
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)

# Define the converltional block 5
x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
x = Dense(12, activation='softmax')(x)

model = Model(inputs= inputs, outputs= x)
model = multi_gpu_model(model, 4) 

# compile the model 
ada = Adam(lr=0.001)
model.compile(optimizer=ada, 
        loss='categorical_crossentropy', 
        metrics=['acc'])

data_dir = ""
person = []
angle = ['0', '45', '90']

all_x = []
all_y = []

for p in person:
    for a in angle:
        path = os.path.join(data_dir, p, a)
        activity = sorted(os.listdir(path))
        for active in activity:
                class_name = int(active[0:2])-1
                dop_file = os.path.join(path, active, "doppler_gt.npy")
                dopler = np.load(dop_file).T
                dopler = get_spectograms(dopler, 3, 24)
                dopler = np.expand_dims(dopler[0:178], axis=-1)
                class_arr = np.array([class_name] * dopler.shape[0])
                all_x.append(dopler)
                all_y.append(class_arr)

all_X = all_x[0]
all_Y = all_y[0]
for i in range(1, len(all_x)):
    all_X = np.concatenate((all_X, all_x[i]), axis=0)
    all_Y = np.concatenate((all_Y, all_y[i]), axis=0)
all_Y = np.expand_dims(np.array(all_Y), axis=-1)
all_Y = np_utils.to_categorical(all_Y, num_classes=12)
all_X = np.kron(all_X, np.ones((1,7,3,1)))
all_X = np.reshape(all_X, (-1, 224, 216, 1))
all_Y = np.reshape(all_Y, (-1, 12))

train_X, train_Y, val_X, val_Y = [], [], [], []
train_rate = 0.8
for i in range(all_X.shape[0]):
    rand = np.random.rand(1)
    if rand < train_rate:
        train_X.append(all_X[i])
        train_Y.append(all_Y[i])
    else:
        val_X.append(all_X[i])
        val_Y.append(all_Y[i])

train_X = np.array(train_X)
train_Y = np.array(train_Y)
val_X = np.array(val_X)
val_Y = np.array(val_Y)

model.fit(train_X, train_Y, validation_data=(val_X, val_Y), batch_size=128, epochs=100)

model.save("save_weights/resnet34_classifier_weights_100.hdf5")