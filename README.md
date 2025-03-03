# Turn That Frown Upside Down: FaceID Customization via Cross-Training Data



This repository contains resources referenced in the paper [Turn That Frown Upside Down: FaceID Customization via Cross-Training Data](https://arxiv.org/abs/2501.15407v1). 

If you find this repository helpful, please cite the following:
```latex
@misc{wang2025turnfrownupsidedown,
      title={Turn That Frown Upside Down: FaceID Customization via Cross-Training Data}, 
      author={Shuhe Wang and Xiaoya Li and Xiaofei Sun and Guoyin Wang and Tianwei Zhang and Jiwei Li and Eduard Hovy},
      year={2025},
      eprint={2501.15407},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.15407}, 
}
```



## 🥳 News 

**Stay tuned! More related work will be updated!**
* **[25 Jan, 2025]** The repository is created. 
* **[25 Jan, 2025]** We release the first version of the paper.


## Links
- [Turn That Frown Upside Down: FaceID Customization via Cross-Training Data](#turn-that-frown-upside-down-faceid-customization-via-cross-training-data)
  - [🥳 News](#-news)
  - [Links](#links)
  - [Introduction](#introduction)
  - [Comparison with Previous Works](#comparison-with-previous-works)
    - [FaceID Fidelity](#faceid-fidelity)
    - [FaceID Customization](#faceid-customization)
  - [Released CrossFaceID dataset](#released-crossfaceid-dataset)
  - [Released FaceCustomization Models](#released-facecustomization-models)
  - [Usage](#usage)
    - [Training](#training)
      - [Step 1. Download Required Models](#step-1-download-required-models)
      - [Step 2. Download Required Dataset](#step-2-download-required-dataset)
      - [Step 3. Training](#step-3-training)
    - [Inference](#inference)
  - [Contact](#contact)



## Introduction

CrossFaceID is the first large-scale, high-quality, and publicly available dataset specifically designed to improve the facial modification capabilities of FaceID customization models. Specifically, CrossFaceID consists of 40,000 text-image pairs from approximately 2,000 persons, with each person represented by around 20 images showcasing diverse facial attributes such as poses, expressions, angles, and adornments. During the training stage, a specific face of a person is used as input, and the FaceID customization model is forced to generate another image of the same person but with altered facial features. This allows the FaceID customization model to acquire the ability to personalize and modify known facial features during the inference stage.

<div align="center">
  <img src="assets/examples.png" width="960">
</div>


## Comparison with Previous Works


### FaceID Fidelity

<div align="center">
  <img src="assets/results_fidelity.png" width="960">
</div>

The results demonstrate the performance of FaceID customization models in maintaining FaceID fidelity. For models, “InstantID” refers to the official InstantID model, while “InstantID + CrossFaceID” represents the model further fine-tuned on our CrossFaceID dataset. “LAION” denotes the InstantID model pre-trained on our curated LAION dataset, and “LAION + CrossFaceID” refers to the model further trained on the CrossFaceID dataset. These results indicate that (1) for both the official InstantID model and the LAION-trained model, the ability to maintain FaceID fidelity remains consistent before and after fine-tuning on our CrossFaceID dataset, and (2) the model trained on our curated LAION dataset achieves comparable performance to the official InstantID model in preserving FaceID fidelity.

### FaceID Customization

<div align="center">
  <img src="assets/results_customization.png" width="960">
</div>

The results of the performance for FaceID customization models in customizing or editing FaceID. Here, "InstantID" represents the official InstantID model, while "InstantID + CrossFaceID" refers to the model fine-tuned on our CrossFaceID dataset. Similarly, "LAION" denotes the InstantID model pre-trained on our curated LAION dataset, and "LAION + CrossFaceID" refers to the model further fine-tuned on the CrossFaceID dataset. From these results, we can clearly observe an improvement in the models' ability to customize FaceID after being fine-tuned on our constructed CrossFaceID dataset.



## Released CrossFaceID dataset

Our CrossFaceID dataset is available on [Huggingface](https://huggingface.co/datasets/Super-shuhe/CrossFaceID). It comprises 40,000 text-image pairs from approximately 2,000 individuals, with each person represented by around 20 images that capture various facial attributes, including different poses, expressions, angles, and adornments.


## Released FaceCustomization Models

The trained InstantID model is available [here](https://huggingface.co/Super-shuhe/CrossFaceID-InstantID).

## Usage

### Training
As the original InstantID repository (https://github.com/InstantID/InstantID) doesn't contain training codes, we follow [this repository](https://github.com/MFaceTech/InstantID?tab=readme-ov-file) to train our own InstantID.

#### Step 1. Download Required Models

You can directly download the model from [Huggingface](https://huggingface.co/InstantX/InstantID).
You also can download the model in python script:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints")
```

If you cannot access to Huggingface, you can use [hf-mirror](https://hf-mirror.com/) to download models.
```python
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download InstantX/InstantID --local-dir checkpoints
```

For face encoder, you need to manutally download via this [URL](https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304) to `models/antelopev2` as the default link is invalid. Once you have prepared all models, the folder tree should be like:

```
  .
  ├── models
  ├── checkpoints
  ├── ip_adapter
  ├── pipeline_stable_diffusion_xl_instantid.py
  ├── download.py
  ├── download.sh
  ├── get_face_info.py
  ├── infer_from_pkl.py
  ├── infer.py
  ├── train_instantId_sdxl.py
  ├── train_instantId_sdxl.sh
  └── README.md
```


#### Step 2. Download Required Dataset

Please download our released dataset from [Huggingface](https://huggingface.co/datasets/Super-shuhe/CrossFaceID).

#### Step 3. Training

1. Fill the `MODEL_NAME`, `ENCODER_NAME`, `ADAPTOR_NAME`, `CONTROLNET_NAME`, and `JSON_FILE` into our provided training script `./train_instantId_sdxl.sh`, where:
   1. `MODEL_NAME` refers to the backboned diffusion model, e.g., `stable-diffusion-xl-base-1.0`
   2. `ENCODER_NAME` refers to the downloaded encoder, e.g., `image_encoder`
   3. `ADAPTOR_NAME` and `CONTROLNET_NAME` refers to the pre-trained official InstantID model, e.g., `checkpoints/ip-adapter.bin` and `checkpoints/ControlNetModel`
   4. `JSON_FILE` refers to our released CrossFaceID dataset.
2. Run the training scirpt, such as: `bash ./train_instantId_sdxl.sh`


### Inference

1. Fill the `base_model_path`, `face_adapter`, `controlnet_path`, `prompt0`, and `face_image` into our provided inference script `./infer_from_pkl.py`, where:
   1. `base_model_path` refers to the backboned diffusion model, e.g., `stable-diffusion-xl-base-1.0`
   2. `face_adapter` and `controlnet_path` refer to your trained model e.g., `checkpoints/ip-adapter.bin` and `checkpoints/ControlNetModel`
   3. `prompt0` and `face_image` refer to your test sample.
2. Run the training script, such as: `python ./infer_from_pkl.py`


## Contact

If you have any issues or questions about this repo, feel free to contact shuhewang@student.unimelb.edu.au
