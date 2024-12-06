# SSSSC

This is an easy-to-use, modular and extensible code repository for semi-supervised semantic segmentation algorithms, packages as well as many core component layers that can be used to easily build repositories of custom models.

## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Mean-teacher  | [NeurIPS 2017][Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://proceedings.neurips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)             |
|     CutMix      | [ICCV 2019][CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features](https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html)                                                   |       
| Fixmatch | [NeurIPS 2020][FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html)                    |
|     Unimatch      | [CVPR 2023][Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)                                                   |            
|     Allspark| [CVPR 2024][AllSpark: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_AllSpark_Reborn_Labeled_Features_from_Unlabeled_in_Transformer_for_Semi-Supervised_CVPR_2024_paper.html)                                                   |          
                                             
## Notice
This repository is intended to provide a convenient learning repository for novice learners, and **some of the implementation details may not be consistent with the original author**. If you need to write the experimental data of the paper, **please take the source repository of each paper as the standard**. Contact me if there are any errors in the implemented code.

### Dataset

```
├── [Your Dataset Path]
    ├── JPEGImages
    └── SegmentationClass
    └── ImageSets
    	└──Segmentation
    		└──split.txt
    		
```
Modify your dataset path in train.py
``
Dataset_path  = 'YourDataset'
`` 

## Ongoing update... ...





