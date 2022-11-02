# SufrinNet: Toward Sufficient Cross-View Interaction for Stereo Image Enhancement in The Dark (arXiv 2022)

<hr />

**Abstract:** *Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, have shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising).* 
<hr />

## Network Architecture

<img src = "https://github.com/Ian0926/SufrinNet/blob/main/files/framwork.PNG"> 
<hr />

## Datasets


## Results


## Commands

### Train:
`python test.py --filepath img_path --pretrain_path model_path`

### Test:
`python test.py --filepath img_path --pretrain_path model_path`

### Evaluation
`python metrics.py --data-path gt_path --output-path pre_img_path`

## Bibtex:
If this repository is useful for you, please consider citing:
```
@inproceedings{zheng2022deep,
  title={SufrinNet: Toward Sufficient Cross-View Interaction for Stereo Image Enhancement in The Dark},
  author={Huan Zheng, Zhao Zhang, Jicong Fan, Richang Hong, Yi Yang, and Shuicheng Yan},
  booktitle={arXiv},
  year={2022}
}
```
