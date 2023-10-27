[![Durham](https://img.shields.io/badge/UK-Durham-blueviolet)](https://dro.dur.ac.uk/38185/)
[![CVF](https://img.shields.io/badge/IEEE-CVF-blue)](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Less_Is_More_Reducing_Task_and_Model_Complexity_for_3D_CVPR_2023_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2303.11203-b31b1b.svg)](https://arxiv.org/abs/2303.11203)
[![GitHub license](https://img.shields.io/badge/license-Apache2.0-blue.svg)](https://github.com/l1997i/lim3d/blob/main/LICENSE)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white&style=flat)
![Stars](https://img.shields.io/github/stars/l1997i/lim3d?style=social)
<div align="center">
    <img src="./img/durham_logo.png" width="15%" />  &emsp;&emsp;&emsp;&emsp;
    <img src="./img/cvpr23_logo.png" width="20%" /> <br>
</div> 


# 🔥 Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation [CVPR 2023]

[Li Li](https://luisli.org), [Hubert P. H. Shum](http://hubertshum.com/) and [Toby P. Breckon](https://breckon.org/toby/), In Proc. International Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2023
[[homepage](https://project.luisli.org/lim3d/)] [[pdf](https://arxiv.org/pdf/2303.11203.pdf)] [[video](https://www.bilibili.com/video/BV1ih4y1b7gp/?share_source=copy_web&vd_source=0f5e46081bb4e59af9a0ccd37c106f35)] [[poster](https://www.luisli.org/assets/pdf/li23cvpr_poster_big.pdf)]



https://github.com/l1997i/lim3d/assets/35445094/d52f9d80-c4dc-4147-af0c-6101ca6f6b0f


> **Abstract**: Whilst the availability of 3D LiDAR point cloud data has significantly grown in recent years, annotation remains expensive and time-consuming, leading to a demand for semi-supervised semantic segmentation methods with application domains such as autonomous driving. Existing work very often employs relatively large segmentation backbone networks to improve segmentation accuracy, at the expense of computational costs. In addition, many use uniform sampling to reduce ground truth data requirements for learning needed, often resulting in sub-optimal performance. To address these issues, we propose a new pipeline that employs a smaller architecture, requiring fewer ground-truth annotations to achieve superior segmentation accuracy compared to contemporary approaches. This is facilitated via a novel Sparse Depthwise Separable Convolution module that significantly reduces the network parameter count while retaining overall task performance. To effectively sub-sample our training data, we propose a new Spatio-Temporal Redundant Frame Downsampling (ST-RFD) method that leverages knowledge of sensor motion within the environment to extract a more diverse subset of training data frame samples. To leverage the use of limited annotated data samples, we further propose a soft pseudo-label method informed by LiDAR reflectivity. Our method outperforms contemporary semi-supervised work in terms of mIoU, using less labeled data, on the SemanticKITTI (**59.5**@5%) and ScribbleKITTI (**58.1**@5%) benchmark datasets, based on a **2.3×** reduction in model parameters and **641×** fewer multiply-add operations whilst also demonstrating significant performance improvement on limited training data (*i.e.*, *Less is More*).

![](./img/pipeline.png)

## News 

[2023/06/21] 🇨🇦 We will **present our work** in West Building Exhibit Halls ABC 108 @ Wed 21 Jun 10:30 a.m. PDT — noon PDT. See you in Vancouver, Canada.  
[2023/06/20] **Code released**.  
[2023/02/27] LiM3D was **accepted** at CVPR 2023!

## Data Preparation

The `data` is organized in the format of `{SemanticKITTI}` U `{ScribbleKITTI}`.

```
sequences/
    ├── 00/
    │   ├── scribbles/
    │   │     ├ 000000.label
    │   │     ├ 000001.label
    │   │     └ .......label
    │   ├── labels/
    │   ├── velodyne/
    │   ├── image_2/
    │   ├── image_3/
    │   ├── times.txt
    │   ├── calib.txt
    │   └── poses.txt
    ├── 01/
    ├── 02/
    .
    .
    └── 21/
```

### SemanticKITTI
Please follow the instructions from [SemanticKITTI](http://www.semantic-kitti.org) to download the dataset including the KITTI Odometry point cloud data.

### ScribbleKITTI
Please download `ScribbleKITTI` [scribble annotations](https://data.vision.ee.ethz.ch/ouenal/scribblekitti.zip) and unzip in the same directory. Each sequence in the train-set (00-07, 09-10) should contain the `velodyne`, `labels` and `scribbles` directories.

Move the `sequences` folder or make a symbolic link to a new directory inside the project dir called `data/`. Alternatively, edit the `dataset: root_dir` field of each config file to point to the sequences folder.

## Environment Setup

For the installation, we recommend setting up a virtual environment using `conda` or `venv`:

For conda,
```shell
conda env create -f environment.yaml
conda activate lim3d 
pip install -r requirements.txt
```

For venv,
```shell
python -m venv ~/venv/lim3d
source ~/venv/scribblekitti/bin/activate
pip install -r requirements.txt
```

Furthermore install the following dependencies:
- [pytorch](https://pytorch.org/get-started/previous-versions/#v1101) (tested with version `1.10.1+cu111`)
- [pytorch-lightning](https://www.pytorchlightning.ai/) (tested with version `1.6.5`)
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter) (tested with version `2.0.9`)
- [spconv](https://github.com/traveller59/spconv) (tested with version `2.1.21`)

## Experiments

Our overall architecture involves three stages (Figure 2). You can reproduce our results through the scripts provided in the `experiments` folder: 

1. **Training**: we utilize reflectivity-prior descriptors and adapt the Mean Teacher framework to generate high-quality pseudo-labels. Running with bash script: `bash experiments/train.sh`; 
2. **Pseudo-labeling**: we fix the trained teacher model prediction in a class-range-balanced manner, expanding dataset with Reflectivity-based Test Time Augmentation (Reflec-TTA) during test time. Running with bash script: `bash experiments/crb.sh`, then save the pseudo-labels `bash experiments/save.sh`; 
3. **Distillation with unreliable predictions**: we train on the generated pseudo-labels, and utilize unreliable pseudo-labels in a category-wise memory bank for improved discrimination. Running with bash script: `bash experiments/distilation.sh`.

## Results

Please refer to our **supplementary 
** and **supplementary documentation** for more qualitative results.

You can download our pretrained models [here](https://durhamuniversity-my.sharepoint.com/:f:/g/personal/mznv82_durham_ac_uk/Es4lmKcQ49lIh57u89gI5UsBWBBSeq-LdbZedfBS9m1x3g?e=3fr3YR) via `Onedrive`.

To validate the results, please refer to the scripts in `experiments` folder, and put the pretrained models in the `models` folder. Specify `CKPT_PATH` and `SAVE_DIR` in `predict.sh` file. 

For example, if you want to validate the results of 10% labeled training frames + LiM3D (without SDSC) + with `reflectivity` features on `ScribbleKITTI`, you can specify `CKPT_PATH` as `model/sck_crb10_feat69_61.01.ckpt`. Run following scripts:

```bash
bash experiments/predict.sh
```


## Citation

If you are making use of this work in any way, you must please reference the following paper in any report, publication, presentation, software release or any other associated materials:

[Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation](#) ([Li Li](https://luisli.org), [Hubert P. H. Shum](http://hubertshum.com/) and [Toby P. Breckon](https://breckon.org/toby/)), In IEEE Conf. Comput. Vis. Pattern Recog. (CVPR), 2023. [[homepage](https://project.luisli.org/lim3d/)] [[pdf](https://arxiv.org/pdf/2303.11203.pdf)] [[video](https://www.bilibili.com/video/BV1ih4y1b7gp/?vd_source=cc0410bc3f69236950fa663b082e6754)] [[poster](https://www.luisli.org/assets/pdf/li23cvpr_poster_big.pdf)]

```bibtex
@InProceedings{li23lim3d,
  title      =    {Less Is {{More}}: {{Reducing Task}} and {{Model Complexity}} for {{3D Point Cloud Semantic Segmentation}}},
  author     =    {Li, Li and Shum, Hubert P. H. and Breckon, Toby P.},
  keywords   =    {point cloud, semantic segmentation, sparse convolution, depthwise separable convolution, autonomous driving},
  year       =    {2023},
  month      =    June,
  publisher  =    {{IEEE}},
  booktitle  =    {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```


---
### Acknowledgements
We would like to additionally thank the authors the open source codebase [ScribbleKITTI](https://github.com/ouenal/scribblekitti), [Cylinder3D](https://github.com/xinge008/Cylinder3D), and [U2PL](https://github.com/Haochen-Wang409/U2PL).
