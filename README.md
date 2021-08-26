# A2GNN
Affinity Attention Graph Neural Network for Weakly Supervised Semantic Segmentation (TPAMI 2021).

This project is built based on the implementation of [SEAM](https://github.com/YudeWang/SEAM), [GatedCRF](https://github.com/LEONOB2014/GatedCRFLoss) and [AGNN](https://github.com/dawnranger/pytorch-AGNN).

Many thanks for their great work.

## Step 1: Generate Seed Label
In this step, we try to generate the initial seed labels from different weak supervised cases.

You can directly download the seed label from [here](https://drive.google.com/file/d/1d6nntO4kfgKN45o37GjTYWdu6eLF8RXc/view?usp=sharing) and put them into ./data folder.

>The whole processing is: 
> 1. Train the classification network (train_SEAM.py in [SEAM](https://github.com/YudeWang/SEAM)) and the trained model is available in [SEAM Google Drive](https://drive.google.com/file/d/1jWsV5Yev-PwKgvvtUM3GnY0ogb50-qKa/view).
> 2. Generate the confident label (top 40%) through using our infer_SEAM.py (compared to original infer_SEAM.py in SEAM, we just add the code to select confident labels). For image-level case, please also save the original CAM file (--out_cam should be given a path).
> 3. For box, scribble and point cases, merge the corresponding labels following our paper. The GrabCut label is generated using the code from [SDI](https://github.com/johnnylu305/Simple-does-it-weakly-supervised-instance-and-semantic-segmentation).

I will update the code of the whole process in the future.

## Step 2: Train Affinity Network
In this step, we try to train a CNN which can convert an image to a graph.

You need to prepare at least 2 GPUs (2080Ti) and download the initial model weights from [SEAM Google Drive](https://drive.google.com/file/d/1jWsV5Yev-PwKgvvtUM3GnY0ogb50-qKa/view) or our [BaiDuYun](https://pan.baidu.com/s/1Y7K8o6FEIqdhAOTKgcA39g) (access code: cvfg).

for the box and scribble supervision:

    python train_aff_gated_aff.py --radius 4 --label_dir ./data/Init_Label/[SEAM_Box, SEAM_Scribble] --weights ./netWeights/resnet38_aff_SEAM.pth --session_name res_aff_box --voc12_root your/path/VOC2012

for the image-level and point supervision:

    python train_aff_gated_aff.py --radius 3 --label_dir ./data/Init_Label/[SEAM_Point, SEAM_Image] --weights ./netWeights/resnet38_aff_SEAM.pth --session_name res_aff_box --voc12_root your/path/VOC2012

For all cases, we used resnet38_aff_SEAM.pth as the initial weights, which is trained using top 100% label from infer_SEAM.py.

## Step 3: Generate the Final Pseudo Labels
All the trained model of Step 2 can be download from our [BaiDuYun](https://pan.baidu.com/s/1Y7K8o6FEIqdhAOTKgcA39g) (access code: cvfg).

For Box supervision:
    
    python train_infer_A2GNN_box.py --voc12_root /your/path/VOC2012 --BoxXmlpath /your/path/VOC2012/Annotations --weights ./netWeights/final_model/aff_box.pth --save_path ./out/box_pred --seed_label_root ./data/Init_Label/SEAM_Box

For Scribble supervision:

    python train_infer_A2GNN_others.py --rw False --voc12_root /your/path/VOC2012 --weights ./netWeights/final_model/aff_scribble.pth --save_path ./out/scibble_pred --seed_label_root ./data/Init_Label/Scribble_SuperPixel
    
For Image and Point supervision:

    python train_infer_A2GNN_others.py --rw True --voc12_root /your/path/VOC2012 --weights ./netWeights/final_model/[aff_image.pth, aff_point.pth] --save_path ./out/[image_pred,point_pred] --seed_label_root ./data/Init_Label/[SEAM_Point, SEAM_Image] --cam_dir /your/path/cam_dir

The CAM files can be generated using "infer_SEAM.py" (set --out_cam=/your/path)

Also, you can directly download them (about 11G) from [Our BaiDuYun](https://pan.baidu.com/s/1TaaR8jPW-QDseZarC0oK0w) (access code:73po).

## Step 4: Train a Segmentation Model
For Deeplab v2, we used the code from [here](https://github.com/kazuto1011/deeplab-pytorch). 

For PSPNet, we used the code from [here](https://github.com/hszhao/semseg).

For Tree-FCN, we used the code from [here](https://github.com/Megvii-BaseDetection/TreeFilter-Torch).

For Mask-RCNN, we used [Detectron2](https://github.com/facebookresearch/detectron2).

To train the semantic segmentation model, please reduce the learning rate following [SC-CAM](https://github.com/Juliachang/SC-CAM/issues/5).

**Note**: for generating the results of our ablation study, please set '--infer_list' in 'train_infer_A2GNN_box' and 'train_infer_A2GNN_others' as 'train.txt'.

## Related Work
[1] Song, Lin, et al. "Learnable tree filter for structure-preserving feature transform." Advances in Neural Information Processing Systems, 2019.

[2] Wang, Yude, et al. "Self-supervised equivariant attention mechanism for weakly supervised semantic segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[3] Obukhov, Anton, et al. "Gated CRF loss for weakly supervised semantic image segmentation." arXiv preprint arXiv:1906.04651 (2019).

[4] Thekumparampil, Kiran K., et al. "Attention-based graph neural network for semi-supervised learning." arXiv preprint arXiv:1803.03735 (2018).
## Cite
```
@article{zhang2021affinity,
  title={Affinity Attention Graph Neural Network for Weakly Supervised Semantic Segmentation},
  author={Zhang, Bingfeng and Xiao, Jimin and Jiao, Jianbo and Wei, Yunchao and Zhao, Yao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```