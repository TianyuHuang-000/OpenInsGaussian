# OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with Context-aware Cross-view Fusion  
### ICCV 2025 Findings 

<h3>
  <strong>Paper(<a href="https://www.arxiv.org/abs/2510.18253">arXiv</a> / <a href="https://openaccess.thecvf.com/content/ICCV2025W/Findings/papers/Huang_OpenInsGaussian_Open-vocabulary_Instance_Gaussian_Segmentation_with_Context-aware_Cross-view_Fusion_ICCVW_2025_paper.pdf">Conference</a>)</strong> 
</h3>

---

## Installation

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

## ToDo List

- [ ] Training code and script
- [ ] Data preprocessing   

---

## Data preprocessing (coming soon)

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