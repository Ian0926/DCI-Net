# [SufrinNet: Toward Sufficient Cross-View Interaction for Stereo Image Enhancement in The Dark]() (arXiv 2022)

<hr />

**Abstract:** *Low-light stereo image enhancement (LLSIE) is a relatively new task to enhance the quality of visually unpleasant stereo images captured in dark conditions. So far, very few studies on deep LLSIE have been explored due to certain challenging issues, i.e., the task has not been well addressed, and current methods clearly suffer from two shortages: 1) insufficient cross-view interaction; 2) lacking long-range dependency for intra-view learning. In this paper, we therefore propose a novel LLSIE model, termed **Suf**ficient C**r**oss-View **In**teraction Network (SufrinNet). To be specific, we present sufficient inter-view interaction module (SIIM) to enhance the information exchange across views. SIIM not only discovers the cross-view correlations at different scales, but also explores the cross-scale information interaction. Besides, we present a spatial-channel information mining block (SIMB) for intra-view feature extraction, and the benefits are twofold. One is the long-range dependency capture to build spatial long-range relationship, and the other is expanded channel information refinement that enhances information flow in channel dimension. Extensive experiments on Flickr1024, KITTI 2012, KITTI 2015 and Middlebury datasets show that our method obtains better illumination adjustment and detail recovery, and achieves SOTA performance compared to other related methods.* 
<hr />

## Network Architecture

<img src = "https://github.com/Ian0926/SufrinNet/blob/main/files/framwork.PNG"> 
<hr />

## Results

<img src = "https://github.com/Ian0926/SufrinNet/blob/main/files/results.PNG"> 

We also share the visual results of **our SufrinNet** and **other compared methods** on four testing sets. You can **download the enhanced images** from [this link](https://share.weiyun.com/Dnq7lKj7)
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
@inproceedings{zheng2022sufrin,
  title={SufrinNet: Toward Sufficient Cross-View Interaction for Stereo Image Enhancement in The Dark},
  author={Huan Zheng, Zhao Zhang, Jicong Fan, Richang Hong, Yi Yang, and Shuicheng Yan},
  booktitle={arXiv},
  year={2022}
}
```
