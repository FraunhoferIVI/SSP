# High Temporal Consistency through Semantic Similarity Propagation in Semi-Supervised Video Semantic Segmentation for Autonomous Flight

This repository includes the training and evaluation code for SSP and KD-SSP, on the UAVid and RuralScapes datasets. No checkpoints are provided at the moment.


## Requirements

A relatively recent version of Python (ex: 3.10) and PyTorch (ex: 2.3) are required. All dependencies can be installed in a virtual environment with ```pip install -r requirements.txt```.


## Datasets

### Downloading UAVid with VDD annotations

UAVid videos can be downloaded at https://uavid.nl/#download, under Semantic Labelling with Video Support.

The VDD annotations can be obtained from https://github.com/RussRobin/VDD.

### Downloading RuralScapes

RuralScapes can be downloaded at https://sites.google.com/site/aerialimageunderstanding/semantics-through-time-semi-supervised-segmentation-of-aerial-videos#h.q8g692kxr62m. 


### Preparing datasets

The datasets zip files should be placed inside the `datasets` folder. The following bash script can then be run to preprocess them for the right folder structure: ```bash process_dataset_sh```. Files and folder names may have to be adjusted inside the script.

This script will prepare both datasets, if only one is needed then the commands related to the other can be commented.

## Configs and saving checkpoints/results

Configs for training are stored in ```config/image``` and ```config/video``` for image segmentation models and SSP/other video methods respectively.

Trained checkpoints and their results will be stored in the ```save_dir``` argument of their config file, default: ```./checkpoints```. Name of their folder will be the date and time of launch. The saved checkpoint corresponds to the last training epoch.


## Train image models (baseline)

For the pre-trained Hiera weights, download the checkpoint ```sam2_hiera_small``` at https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-2-checkpoints and place it in ```base_checkpoints/```.

With pre-written configs:

- UAVid: ```python -m training.train_image base_uavid.yaml```
- RuralScapes: ```python -m training.train_image base_rural.yaml```

See ```config/image/config_base.yaml``` for an explanation of the image model config file.

## Train SSP

Training SSP requires an image model checkpoint (trained or not). To obtain an untrained image model:
- UAVid: ```python -m training.train_image untrained_uavid.yaml```
- RuralScapes: ```python -m training.train_image untrained_rural.yaml```

The name of the image checkpoint must be written in the config of SSP, under the ```image_model_cfg.checkpoint_name``` argument. The ```image_model_cfg.image_save_dir``` field indicates where this checkpoint is found. This applies to all video configs.

To train SSP:
- UAVid: ```python -m training.train_video ssp_uavid.yaml```
- RuralScapes: ```python -m training.train_video ssp_rural.yaml```


## Train teacher model

For the pre-trained Hiera weights, download the checkpoint ```sam2_hiera_base_plus``` at https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-2-checkpoints and place it in ```base_checkpoints/```.

### Training
With pre-written configs:

- UAVid: ```python -m training.train_image config_teacher.yaml```
- RuralScapes: ```python -m training.train_image config_rural_teacher.yaml```

### Save prediction for knowledge distillation

Replace ```CHECKPOINT``` with the name of your teacher model checkpoint. If not using the default `save_dir` (`./checkpoints`), either change the default argument in `eval.vis.image` or add the argument `--save-dir` with the corresponding directory.

```python -m eval.vis.image CHECKPOINT --save-logits --split train --skip-frames 2 --no-evaluation```


## Train KD-SSP

Fill the `data_cfg.logits_folder` argument of the following configs with the path of the folder containing the train logits of the teacher model (the folder should be named `train_logits`, ex: `checkpoints/TEACHER_CHECKPOINT/train_logits`).

- ```python -m training.train_video kd_ssp_uavid.yaml```
- ```python -m training.train_video kd_ssp_rural.yaml```

Train the image baseline with KD:

- ```python -m training.train_image kd_base_uavid.yaml```
- ```python -m training.train_image kd_base_rural.yaml```

## Train Other video models

`python -m training.train_video CONFIG`

With `CONFIG` from:
- UAVid:
    - `dff_uavid.yaml`
    - `netwarp_uavid.yaml`
    - `tcbppm_uavid.yaml`
    - `tcbocr_uavid.yaml`
- RuralScapes:
    - `dff_rural.yaml`
    - `netwarp_rural.yaml`
    - `tcbppm_rural.yaml`
    - `tcbocr_rural.yaml`

## Evaluate model

Given a trained checkpoint with name `CHECKPOINT`:

### Only compute metrics

- Image model: `python -m eval.vis.image CHECKPOINT --no-write-res`
- SSP or other video model: `python -m eval.vis.video CHECKPOINT --no-write-res`

### With visualization
This will write results to disk.

- Image model: `python -m eval.vis.image CHECKPOINT`
- SSP or other video model: `python -m eval.vis.video CHECKPOINT`


## Ablation Study

See the respective config files to view the different arguments:

- Consistency loss weight $\lambda$: `python -m training.train_video lambda_X.yaml`, replace X with $\lambda$ for values in the ablation study
- Cosine similarity interpolation: `python -m training.train_video ablation_cossim.yaml`
- No registration: `python -m training.train_video ablation_registration.yaml`
- No interpolation: `python -m training.train_video ablation_interpolation.yaml`
- No consistency loss: `python -m training.train_video ablation_consistencyloss.yaml`
- Untrained image model: `python -m training.train_video untrainedimage.yaml`, fill `image_model_cfg.checkpoint_name` with an untrained checkpoint