# DISP6D: Disentangled Implicit Shape and Pose Learning for Scalable 6D Pose Estimation

Original implementation of the paper Yilin Wen, Xiangyu Li, Hao Pan, Lei Yang, Zheng Wang, Taku Komura and Wenping Wang, "DISP6D: Disentangled Implicit Shape and Pose Learning for Scalable 6D Pose Estimation", ECCV, 2022. [[paper|supplementary]](https://arxiv.org/abs/2107.12549)

## Requirements
### Environment
The code is tested with the following environment:
```  
Ubuntu 16.04
python 3.6 or 3.7
tensorflow 1.15.0
sonnet 1.23
```

### Pretrained Model
Our pretrained checkpoint files for different settings, and other related data for running the demo code of the inference stage can be downloaded via the following link:
[[Inference Data]](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/ylwen_connect_hku_hk/Ek9cLvaMMJtFvnwluSufX4QBYx7ZiacZ3cF0b0XNcdhYJA?e=MzNOE4)

which includes:
1) ```./ckpts/```: The pretrained ckpt files for Ours-per(Setting I) and Ours-all(Setting III) that are trained on the synthetic CAMERA dataset, and for Setting II that is trained on synthetic images of the first 18 T-LESS objects.
2) ```./demo_data/```: Demo test images and their 2D detection results from T-LESS and REAL275.
3) ```./embeddings/```: Reference rotations for the inference stage of the three settings, with pose codebook and 2D bounding boxes for all 30 T-LESS objects of Setting II.
4) ```./real275_curve/```: .pkl files for visualizing our average precisions at different rotation/translation error and 3D IoU thresholds on REAL275.

You may keep the downloaded ```ws``` folder under the root directory of this git repository.



## Quick Start
### Demo on REAL275 (Setting I & Setting III)
#### Setting I (Ours-per)
Run:
```
python demo_real275.py --trained_category <test_category> --test_category <test_category> --demo_img <img_id>
```

to estimate pose for objects with the specified category ```<test_category> ``` on the specified demo image of ```<img_id>```. Note that in this setting, we refer to the model that is trained solely on this specified category:

#### Setting III (Ours-all)
Run by setting ```--trained_category``` as ```all```:
```
python demo_real275.py --trained_category all --test_category <test_category> --demo_img <img_id>
```

to estimate pose for objects with the specified category ```<test_category> ``` on the specified demo image of ```<img_id>```. Here we refer the model that is trained on a combination of all six categories involved in the CAMERA and REAL275 dataset. 

#### Plot Curves of Pose Evaluation
Run:
```
python draw_curves_real275.py
```
to plot the curves of pose evaluation for Ours-per and Ours-all on REAL275, with regard to metrics of rotation/translation error and 3D IoU.



### Demo on T-LESS (Setting II)
Run:
```
python demo_tless.py --test_obj_id <test_obj_id>
```
to estimate pose for TLESS object with the specified id ```<test_obj_id>``` on the demo image. Our model is trained only on the first 18 T-LESS objects.


## Training

Run ```python train.py``` with parsed arguments to train a network with regard to your training data. 


## Acknowledgement

We relied on the code from [AAE](https://github.com/DLR-RM/AugmentedAutoencoder) and [StyleGAN](https://github.com/NVlabs/stylegan) for the autoencoder framework, and [Multipath-AAE](https://github.com/DLR-RM/AugmentedAutoencoder/tree/multipath) for data processing and augmentation. We also adopt utilities from the [SIXD Toolkit](https://github.com/thodan/sixd_toolkit)


## Citiation
If you find this work helpful, please consider citing
```
@article{wen2022disp6d,
  title={DISP6D: Disentangled Implicit Shape and Pose Learning for Scalable 6D Pose Estimation},
  author={Wen, Yilin and Li, Xiangyu and Pan, Hao and Yang, Lei and Wang, Zheng and Komura, Taku and Wang, Wenping},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year= {2022},
}
```
