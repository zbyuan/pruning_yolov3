# YOLOv3-training-prune

## 环境

Python3.6, 
Pytorch 1.1及以上，
numpy>1.16，
tensorboard=1.13以上

YOLOv3 的训练参考[博客]（https://blog.csdn.net/qq_34795071/article/details/90769094  ） 代码基于的 [PyTorch-YOLOv3](https://github.com/ultralytics/yolov3)


## 正常训练（Baseline）

```bash
python train.py --data data/VHR.data --cfg cfg/yolov3.cfg --weights/yolov3.weights --epochs 100 --batch-size 32 #后面的epochs自行更改 直接加载weights可以更好的收敛
```

## 剪枝算法介绍

本代码基于论文 [Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) 进行改进实现的 channel pruning算法，类似的代码实现还有这个 [yolov3-network-slimming](https://github.com/talebolano/yolov3-network-slimming)。原始论文中的算法是针对分类模型的，基于 BN 层的 gamma 系数进行剪枝的。

以下只是算法的大概步骤，具体实现过程中还要做 s 参数的尝试或者需要进行迭代式剪枝等。


 ## 进行稀疏化训练

```bashpython 
python train.py --cfg cfg/yolov3.cfg --data data/VHR.data --weights weights/last.pt --epochs 100 --batch-size 32 -sr --s 0.0001  #scale参数默认0.001，在数据分布广类别多的或者稀疏时掉点厉害的适当调小s
```
## 训练过程中模型可视化
```bash
tensorboard --logdir=runs 
```
##  模型剪枝
```bash
python prune.py --cfg cfg/yolov3.cfg --data data/VHR.data --weights weights/last.pt --percent 0.5
```

##  模型进行微调
 ```bash
 python train.py --cfg cfg/prune_0.5_yolov3_cfg.cfg --data data/VHR.data --weights weights/prune_0.5_last.weights --epochs 100 --batch-size 32
```
## convert cfg/pytorch model to darknet weights
```bash
python  -c "from models import *; convert('cfg/yolov3.cfg', 'weights/yolov3.pt')"
Success: converted 'weights/yolov3.pt' to 'converted.weights'
```
## 参考
https://github.com/Lam1360/YOLOv3-model-pruning

https://github.com/ultralytics/yolov3
