# Midas: Generating mmWave Radar Data from Videos for Training Pervasive and Privacy-preserving Human Sensing Tasks
## Get Started
Use the links below to download the:
* [vgg16 classifier weight](https://www.dropbox.com/s/ubxg7qewzwv6n2h/classifier_weights.hdf5?dl=0) 
* [encoder-decoder weight](https://www.dropbox.com/s/0kki7buj2vcvnx1/autoencoder_weights.hdf5?dl=0) 
* [real world Doppler dataset](https://www.dropbox.com/s/y15pmjn5erlegpt/radar_data.zip?dl=0) 
* [sample Videos](https://www.dropbox.com/s/b0dpyv8mvo2axfn/video_data.zip?dl=0) 

Folder description：
* `doppler_data/` used to store millimeter Doppler signals,
* `mmw_preprocessing/` used to store millimeter wave signal processing files, 
* `models/` used to store weight files, 
* `video_data/` used to store video files.

File description：
* `train_encoder_decoder.py` used to train encoder-decoder, 
* `train_resnet34_classifier.py` used to train resnet34 classifier, 
* `train_vgg16_classifier.py` used to train vgg16 classifier.


## Dataset
Use the links below to download the:
* [Real world Doppler dataset](https://www.dropbox.com/s/y15pmjn5erlegpt/radar_data.zip?dl=0) collected across 10 participants and 12 activities. Details found in paper here. 
* [Sample Videos](https://www.dropbox.com/s/b0dpyv8mvo2axfn/video_data.zip?dl=0) for demo purposes.

Dataset directory structure：
```
video_data/
|----Participant_01/
|    |----angle_0/
|         |----action_01
|              |----sample_video.mp4
|         |----action_02
|              |----sample_video.mp4
|         |----...
|    |----angle_45/
|         |----...
|    |----angle_90/
|         |----...
|----...

radar_data/
|----Participant_01/
|    |----angle_0/
|         |----action_01
|              |----doppler_gt.npy
|         |----action_02
|              |----doppler_gt.npy
|         |----...
|    |----angle_45/
|         |----...
|    |----angle_90/
|         |----...
|----...
```	
Data is divided into two parts, one is `radar_data/`, which contains radar data. The other part is `video_data/`, which contains video data, which we've only partially disclosed.
## Reference
...
[Download paper here.](...)

BibTex Reference:
```
...
```

