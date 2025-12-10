# OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with Context-aware Cross-view Fusion  
### ICCV 2025 Findings 

<h3>
  <strong>Paper(<a href="https://www.arxiv.org/abs/2510.18253">arXiv</a> / <a href="https://openaccess.thecvf.com/content/ICCV2025W/Findings/papers/Huang_OpenInsGaussian_Open-vocabulary_Instance_Gaussian_Segmentation_with_Context-aware_Cross-view_Fusion_ICCVW_2025_paper.pdf">Conference</a>)</strong> 
</h3>

---

## 1. Installation

The installation follows the standard setup of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

```bash
git clone https://github.com/TianyuHuang-000/OpenInsGaussian
cd OpenInsGaussian
```

### Create conda environment

```bash
conda env create --file environment.yml
conda activate openinsgs
```

### Install rasterization library (from DreamGaussian)

```bash
cd submodules
unzip ashawkey-diff-gaussian-rasterization.zip
pip install ./ashawkey-diff-gaussian-rasterization
```

### Additional dependencies

```bash
pip install bitarray scipy
```

Install a matching PyTorch3D version based on your PyTorch + CUDA.

`simple-knn` is **not required**.

---

## 2.Data preparation
The files are as follows:
```
[DATA_ROOT]
├── [1] scannet/
│   │   ├── scene0000_00/
|   |   |   |── color/
|   |   |   |── language_features/
|   |   |   |── points3d.ply
|   |   |   |── transforms_train/test.json
|   |   |   |── *_vh_clean_2.labels.ply
│   │   ├── scene0062_00/
│   │   └── ...
├── [2] lerf/
│   │   ├── figurines/ & ramen/ & teatime/ & waldo_kitchen/
|   |   |   |── images/
|   |   |   |── language_features/
|   |   |   |── sparse/
│   │   ├── label/
```
+ **[1] Prepare ScanNet Data**
    + You can directly download our pre-processed data: [**OneDrive**](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/thua0588_uni_sydney_edu_au/IQD8jxvBH_4FRr3R-B1gOi5XAQz2AifOlBKntijL3zF7c4Q?download=1).
    + The ScanNet dataset requires permission for use, following the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to apply for dataset permission.
    + The preprocessing script will be updated later.
+ **[2] Prepare lerf_ovs Data**
    + You can directly download our pre-processed data: [**OneDrive**](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/thua0588_uni_sydney_edu_au/IQCZSXlehQMvQLZsgFE_D5NfAZty5eeS-2RGD2-xHspGN_M?download=1) (re-annotated by LangSplat). 


+ **Mask and Language Feature Extraction Details**
    + We follow the preprocessing tools provided by LangSplat to extract SAM masks and CLIP features.
      If you would like to process a custom dataset, please first complete the data preparation steps described in the LangSplat repository, and then run the `preprocess.sh` script inside the fc-clip submodule.

---
## 3. Training

### 3.1 ScanNet

```bash
chmod +x scripts/train_scannet.sh
./scripts/train_scannet.sh
```

- Please review the script for additional arguments and update the dataset paths as needed.
- During training, the pipeline will automatically progress through the following stages:

```text
[Stage 0] Initial 3D Gaussian Splatting pretraining (0–30k steps)
[Stage 1] Instance-level feature refinement (30–50k steps)
[Stage 2.1] Coarse codebook quantization (50–70k steps)
[Stage 2.2] Fine-grained codebook quantization (70–90k steps)
[Stage 3] Alignment between 2D language features and 3D clusters (~1 min)
```

- Outputs from each stage are saved under `***/train_process/stage*/`.
  (For Stage 3, we recommend visualizing results using the LeRF dataset.)

---

### 3.2 LeRF_ovs

```bash
chmod +x scripts/train_lerf.sh
./scripts/train_lerf.sh
```

- Please inspect the script and configure the dataset path before running.
- The training schedule includes the following phases:

```text
[Stage 0] 3DGS warm-up training (0–30k steps)
[Stage 1] Instance feature learning (30–40k steps)
[Stage 2.1] Coarse codebook construction (40–50k steps)
[Stage 2.2] Fine codebook construction (50–70k steps)
[Stage 3] 2D–3D semantic feature association (~1 min)
```

- Intermediate checkpoints are stored in `***/train_process/stage*/`.

---

### 3.3 Custom Data

```bash
chmod +x scripts/train_custom.sh
./scripts/train_custom.sh
```

- Modify paths in the script before running.

## 4. Rendering and Evaluation

### 4.1 Rendering 2D Feature Maps

You can render 2D instance feature maps using the same procedure as
standard 3DGS color rendering:

``` bash
python render.py -m "output/xxxxxxxx-x"
```

The generated feature maps will be stored in the subdirectories\
`renders_ins_feat1` and `renders_ins_feat2`.

------------------------------------------------------------------------

### 4.2 ScanNet Evaluation

*(Open-Vocabulary Point Cloud Understanding)*

You can evaluate text-driven semantic segmentation on ScanNet under the
19-, 15-, and 10-category settings.

Before running the script:

1.  Ensure that `gt_file_path` and `model_path` in the script are
    correctly set.\
2.  Set `target_id` to either **19**, **15**, or **10**, depending on
    the evaluation protocol.

Run:

``` bash
python scripts/eval_scannet.py
```

------------------------------------------------------------------------

### 4.3 LeRF Evaluation

*(Open-Vocabulary Object Selection in 3D Scenes)*

#### (1) Rendering text-selected 3D Gaussians into multi-view images

#### (2) Computing the LeRF evaluation metrics

Run:

``` bash
bash scripts/evaluate_lerf.sh
```
---

## Acknowledgements

This project builds upon the excellent open-source work from the 3D vision and 3DGS community.  
We sincerely thank the authors of:

- [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting)  
- [LangSplat](https://github.com/minghanqin/LangSplat)  
- [LEGaussians](https://github.com/buaavrcg/LEGaussians)  
- [SAGA](https://github.com/Jumpat/SegAnyGAussians)  
- [SAM](https://segment-anything.com/)  
- [OpenGaussian](https://github.com/yanmin-wu/OpenGaussian.git)  
- [FC-CLIP](https://github.com/bytedance/fc-clip.git)

Our implementation is primarily developed based on **OpenGaussian**, with further extensions for instance-level open-vocabulary reasoning and improved multi-view consistent feature learning.

---

## Citation

```
@inproceedings{huang2025openinsgaussian,
  title={OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with Context-aware Cross-view Fusion},
  author={Huang, Tianyu and Chen, Runnan and Hu, Dongting and Huang, Fengming and Gong, Mingming and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6341--6350},
  year={2025}
}
```

---

## 7. Contact
If you have any questions about this project, please feel free to contact Tianyu Huang: thua0588[AT]uni.sydney.edu.au