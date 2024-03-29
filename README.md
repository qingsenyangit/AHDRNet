# AHDRNet
<small> Attention-guided Network for Ghost-free High Dynamic Range Imaging (AHDR)

Qingsen Yan*, Dong Gong*, Qinfeng Shi, Anton van den Hengel, Chunhua Shen, Ian Reid, Yanning Zhang. 
In IEEE Conference on Compute rVision and Pattern Recognition (CVPR), 2019:1751-1760. (\* Equall contribution)
\[[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yan_Attention-Guided_Network_for_Ghost-Free_High_Dynamic_Range_Imaging_CVPR_2019_paper.pdf)\]\[[Project](https://qingsenyangit.github.io/project/ahdr/)\]

<img src='imgs/frame.jpg' width=790>  


## Requirements
+ Python 2.7
+ PyTorch 0.3.1 (tested with 0.3.1)
+ MATLAB (for data preparation)


## Usage
### Data preparation
1. Download data from \[[dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)\]
2. Move the dataset into `./GenerH5Data/TrainingData`. 
3. Run `./GenerH5Data/PrepareData.m`
### Testing
1. Install this repository and the required packages. A pretrained model is in `./trained-model`.
2. Prepare dataset.
   1) Download dataset.
   2) Move the dataset into `./dataset`. 
   3) Processed dataset can be obtained by running the corresponding script in `./GenerH5Data/PrepareData.m`.
3. Run `python script_testing.py` files. 

### Training
1. Prepare dataset.
   1) Download dataset.
   2) Move the dataset into `./dataset`. 
   3) Processed dataset can be obtained by running the corresponding script in `./GenerH5Data/PrepareData.m`.
3. Run `python script_training.py` files. 

### Examples of the Results
<img src='imgs/fig.jpg' width=420> 

### Examples of the Estimated Attention Maps
<img src='imgs/att_map.jpg' width=420> 


## Citation
If you use this code for your research, please cite our paper.

```
@article{yan2021dual,
  title={Dual-attention-guided network for ghost-free high dynamic range imaging},
  author={Yan, Qingsen and Gong, Dong and Shi, Javen Qinfeng and van den Hengel, Anton and Shen, Chunhua and Reid, Ian and Zhang, Yanning},
  journal={International Journal of Computer Vision},
  pages={1--19},
  year={2021},
  publisher={Springer}
}
@article{yan2019attention,
  title={Attention-guided Network for Ghost-free High Dynamic Range Imaging},
  author={Yan, Qingsen and Gong, Dong and Shi, Qinfeng and Hengel, Anton van den and Shen, Chunhua and Reid, Ian and Zhang, Yanning},
  journal={IEEE Conference on Compute rVision and Pattern Recognition (CVPR)},
  year={2019}
  pages={1751-1760}
}
```














