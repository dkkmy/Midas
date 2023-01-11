import numpy as np
from tensorflow.keras.models import load_model
from helper import get_spectograms, root_mean_squared_error
from tensorflow.keras.optimizers import Adam
import os

model_path = "models/"
autoencoder = load_model(model_path+"autoencoder_weights.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})

ada = Adam(lr=0.001)
autoencoder.compile(
    optimizer = ada,
    loss = root_mean_squared_error,
    metrics=['mae', 'acc'],
)

TIME_CHUNK = 3
fps = 24

synth_dir = "synth"
gt_dir = "gt"

person = []
angle = ['0', '45', '90']

gt_list = []
synth_list = []

for p in person:
    for a in angle:
        activity = sorted(os.listdir(os.path.join(synth_dir, p, a)))
        for active in activity:
            real_data = np.load(os.path.join(gt_dir, p, active, "doppler_gt.npy"))
            synth_data = np.load(os.path.join(synth_dir, p, active, "synth.npy"))
            dopler1 = get_spectograms(real_data.T, TIME_CHUNK, fps)
            dopler1 = np.expand_dims(dopler1, axis=-1)
            dopler1 = (dopler1 - np.min(dopler1)) / (np.max(dopler1) - np.min(dopler1))
            dopler1 = dopler1.astype("float32")
            dopler2 = get_spectograms(synth_data, TIME_CHUNK, fps)
            dopler2 = np.expand_dims(dopler2, axis=-1)
            dopler2 = (dopler2 - np.min(dopler2)) / (np.max(dopler2) - np.min(dopler2))
            dopler2 = dopler2.astype("float32")

            gt_list.append(dopler1)
            synth_list.append(dopler2)

gt_list = np.array(gt_list).reshape((-1,32,72,1))
synth_list = np.array(synth_list).reshape((-1,32,72,1))

autoencoder.fit(synth_list, gt_list, batch_size=128, epochs=1000, shuffle=True)
autoencoder.save("save_weights/autoencoder.hdf5")