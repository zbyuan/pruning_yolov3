# YOLOv3-training-prune

## 环境

Python3.6, 
Pytorch 1.0及以上
numpy>1.16
tensorboard=1.13以上

YOLOv3 的训练参考[博客]（https://blog.csdn.net/qq_34795071/article/details/90769094）， 代码基于的 [PyTorch-YOLOv3](https://github.com/ultralytics/yolov3)


## 正常训练（Baseline）

```bash
python train.py --data data/VHR.data --cfg cfg/yolov3.cfg --weights/yolov3.weights --epochs 100 #后面的epochs自行更改 直接加载weights可以更好的收敛
```

## 剪枝算法介绍

本代码基于论文 [Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) 进行改进实现的 channel pruning算法，类似的代码实现还有这个 [yolov3-network-slimming](https://github.com/talebolano/yolov3-network-slimming)。原始论文中的算法是针对分类模型的，基于 BN 层的 gamma 系数进行剪枝的。

以下只是算法的大概步骤，具体实现过程中还要做 s 参数的尝试或者需要进行迭代式剪枝等。


 ## 进行稀疏化训练

```bashpython 
python train.py --data data/VHR.data --cfg cfg/yolov3.cfg --weights/yolov3.weights -sr --s 0.0001 --epochs 100 
```
## 训练过程中模型可视化
```bash
tensorboard --logdir=runs 
```
## 通过U版本的训练后得到pt文件，转化为weights

### convert cfg/pytorch model to darknet weights
```bash
python3  -c "from models import *; convert('cfg/yolov3.cfg', 'weights/yolov3.pt')"
Success: converted 'weights/yolov3.pt' to 'converted.weights'
```
##  将转化好的weights放到yolo_pruning中weights中进行模型剪枝，基于test_prune.py 文件进行剪枝，得到剪枝后的模型  得到剪枝后的cfg与weights 对剪枝后的模型进行微调
   ```bash
   python train.py --data data/VHR.data --cfg cfg/yolov3_0.85.cfg --weights/yolov3_0.85.weights --epochs 100
   ```
