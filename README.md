# Multi-domain Evaluation of Semantic Segmentation (MESS) with OVSeg

[[Website (soon)](https://github.io)] [[arXiv (soon)](https://arxiv.org/)] [[GitHub](https://github.com/blumenstiel/MESS)]

This directory contains the code for the MESS evaluation of OVSeg. Please see the commits for our changes of the model.

## Setup
Create a conda environment `ovseg` and install the required packages. See [mess/README.md]([mess/README.md]) for details.
```sh
 bash mess/setup_env.sh
```

Prepare the datasets by following the instructions in [mess/DATASETS.md](mess/DATASETS.md). The `ovseg` env can be used for the dataset preparation. If you evaluate multiple models with MESS, you can change the `dataset_dir` argument and the `DETECTRON2_DATASETS` environment variable to a common directory (see [mess/DATASETS.md](mess/DATASETS.md) and [mess/eval.sh](mess/eval.sh), e.g., `../mess_datasets`). 

Download the OVSeg weights (see https://github.com/facebookresearch/ov-seg/blob/main/GETTING_STARTED.md)
```sh
mkdir weights
conda activate ovseg
# Python code for downloading the weights from GDrive. Link: https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view
python -c "import gdown; gdown.download(f'https://drive.google.com/uc?export=download&confirm=pbef&id=1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy', output='weights/ovseg_swinbase_vitL14_ft_mpt.pth')"
```

## Evaluation
To evaluate the OVSeg model on the MESS dataset, run
```sh
bash mess/eval.sh

# for evaluation in the background:
nohup bash mess/eval.sh > eval.log &
tail -f eval.log 
```

For evaluating a single dataset, select the DATASET from [mess/DATASETS.md](mess/DATASETS.md), the DETECTRON2_DATASETS path, and run
```
conda activate ovseg
export DETECTRON2_DATASETS="datasets"
DATASET=<dataset_name>

# OVSeg large model
python train_net.py --num-gpus 1 --eval-only --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml MODEL.WEIGHTS weights/ovseg_swinbase_vitL14_ft_mpt.pth OUTPUT_DIR output/OVSeg/$DATASET DATASETS.TEST \(\"$DATASET\",\)
```

# --- Original OVSeg README.md ---

# [OVSeg] Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP

<img src="resources/pytorch-logo-dark.png" width="10%">

This is the official PyTorch implementation of our paper: <br>
**Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP**<br>
[Feng Liang](https://jeff-liangf.github.io/), [Bichen Wu](https://www.linkedin.com/in/bichenwu), [Xiaoliang Dai](https://sites.google.com/view/xiaoliangdai/), [Kunpeng Li](https://kunpengli1994.github.io/), [Yinan Zhao](https://yinan-zhao.github.io/), [Hang Zhang](https://hangzhang.org/), [Peizhao Zhang](https://www.linkedin.com/in/peizhao-zhang-14846042/), [Peter Vajda](https://sites.google.com/site/vajdap), [Diana Marculescu](https://www.ece.utexas.edu/people/faculty/diana-marculescu) <br>
Computer Vision and Pattern Recognition Conference (CVPR), 2023

[[arXiv](https://arxiv.org/abs/2210.04150)] [[Project](https://jeff-liangf.github.io/projects/ovseg/)] [[huggingface demo](https://huggingface.co/spaces/facebook/ov-seg)]

<p align="center">
<img src="resources/ovseg.gif" width="100%">
</p>


## Installation    

Please see [installation guide](./INSTALL.md).

## Data Preparation

Please see [datasets preparation](./datasets/DATASETS.md).

## Getting started

Please see [getting started instruction](./GETTING_STARTED.md).

## Finetuning CLIP

Please see [open clip training](./open_clip_training/README.md).

## LICENSE

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

The majority of OVSeg is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

However portions of the project are under separate license terms: CLIP and ZSSEG are licensed under the [MIT license](https://github.com/openai/CLIP/blob/main/LICENSE); MaskFormer is licensed under the [CC-BY-NC](https://github.com/facebookresearch/MaskFormer/blob/main/LICENSE); openclip is licensed under the license at [its repo](https://github.com/mlfoundations/open_clip/blob/main/LICENSE).


## Citing OVSeg :pray:

If you use OVSeg in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry.

```BibTeX
@inproceedings{liang2023open,
  title={Open-vocabulary semantic segmentation with mask-adapted clip},
  author={Liang, Feng and Wu, Bichen and Dai, Xiaoliang and Li, Kunpeng and Zhao, Yinan and Zhang, Hang and Zhang, Peizhao and Vajda, Peter and Marculescu, Diana},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7061--7070},
  year={2023}
}
```
