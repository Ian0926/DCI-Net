# [Decoupled Cross-Scale Cross-View Interaction for Stereo Image Enhancement in The Dark](https://arxiv.org/abs/2211.00859)

<hr />

**Abstract:** *Low-light stereo image enhancement (LLSIE) is a relatively new task to enhance the quality of visually unpleasant stereo images captured in dark condition. However, current methods achieve inferior performance on detail recovery and illumination adjustment. We find it is because: 1) the insufficient single-scale inter-view interaction makes the cross-view cues unable to be fully exploited; 2) lacking long-range dependency leads to the inability to deal with the spatial long-range effects caused by illumination degradation. To alleviate such limitations, we propose a LLSIE model termed Decoupled Cross-scale Cross-view Interaction Network (DCI-Net). Specifically, we present a decoupled interaction module (DIM) that aims for sufficient dual-view information interaction. DIM decouples the dual-view information exchange into discovering multi-scale cross-view correlations and further exploring cross-scale information flow. Besides, we present a spatial-channel information mining block (SIMB) for intra-view feature extraction, and the benefits are twofold. One is the long-range dependency capture to build spatial long-range relationship, and the other is expanded channel information refinement that enhances information flow in channel dimension. Extensive experiments on Flickr1024, KITTI 2012, KITTI 2015 and Middlebury datasets show that our method obtains better illumination adjustment and detail recovery, and achieves SOTA performance compared to other related methods. Our codes, datasets and models will be publicly available.* 
<hr />

## Network Architecture

<img src = "https://github.com/Ian0926/SufrinNet/blob/main/files/framwork.PNG"> 
<hr />

## Results

<img src = "https://github.com/Ian0926/SufrinNet/blob/main/files/result.PNG"> 

**Download the enhanced images (ZeroDCE, iPASSRNet, ZeroDCE++, DVENet, NAFSSR, NAFSSR-F, SNR and our method)** from [this link](https://share.weiyun.com/Dnq7lKj7)
<hr />

## Train and Evaluation
* **Dataset and Pretrained Model**:
The synthesized **LLSIE dataset** can be downloaded from [this link](https://share.weiyun.com/Q8nhiqnh) and **pretrained model** can be found [here](https://github.com/Ian0926/SufrinNet/tree/main/pretrained)

* **Train**:
`python train.py --data_source your_data_path`

* **Infer (obtain visual results)**:
`python infer.py --data_source your_data_path --model_path your_model_path --save_image`

* **Test (obtain numerical results)**：
`python test.py --data_source your_data_path --model_path your_model_path`
<hr />

## Bibtex:
If this repository is useful for you, please consider citing:
```
@inproceedings{zheng2022dcinet,
  title={Decoupled Cross-Scale Cross-View Interaction for Stereo Image Enhancement in The Dark},
  author={Huan Zheng, Zhao Zhang, Jicong Fan, Richang Hong, Yi Yang, and Shuicheng Yan},
  booktitle={arXiv},
  year={2022}
}
```
