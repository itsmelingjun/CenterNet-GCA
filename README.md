#   CenterNet-GCA:CenterNet for gravel clusters analysis
![Alt Text](CenterNet-GCA.png)
##   Environment
torch==1.13.0
##   Trained Weights
The pre-trained weights UD00.pth, UD00H1.pth, and UDH00.pth, trained on natural pebble and Hassan laboratory pebble datasets, can be downloaded from Baidu Netdisk.
link: https://pan.baidu.com/s/1rMWSSVxKAHcL5FRsB2YDtg
Extraction code: 8x95
##   Instructions
###   使用CenterNet-GCA对卵石图片进行检测
1.  下载权重，并将其存放地址修改写入newmask_CenterNet文件中的model_path参数。
2.  根据需要，修改newmask_predict文件中的mode参数。若选择dir_predict模式，还需修改dir_origin_path、dir_save_path参数。
3.  可根据检测效果微调newmask_CenterNet文件中的confidence参数，建议从0.2开始调整。
###   使用CenterNet-GCA训练新的卵石数据集
1.  根据需要，修改newmask_train文件中的相关训练参数。
2.  微调训练时需设置pretrained参数为true，并将已有的权重写入model_path，推荐使用UD00作为初始权重。
##   Example
The following detection results were obtained using the UD00 weights.
![Alt Text](img_out/1.JPG)
![Alt Text](img_out/6.JPG)
##   Attention
1.  Improvements were made only for the case where the backbone network is ResNet50; in the code, all backbone parameters support only resnet50.
2.  When detecting large-size images, they should first be cropped into smaller images (preferably no larger than 1500×1000) before being processed by CenterNet-GCA.
##   Reference
-  https://github.com/bubbliiiing/centernet-pytorch
-  https://github.com/xingyizhou/CenterNet

##   所需环境
torch==1.13.0
##   权重下载
已在天然卵石和Hassan实验室卵石数据集上训练好的权重UD00.pth、UD00H1.pth和UDH00.pth可在百度网盘中下载。
链接: https://pan.baidu.com/s/1rMWSSVxKAHcL5FRsB2YDtg
提取码: 8x95
##   使用说明
###   使用CenterNet-GCA对卵石图片进行检测
1.  下载权重，并将其存放地址修改写入newmask_CenterNet文件中的model_path参数。
2.  根据需要，修改newmask_predict文件中的mode参数。若选择dir_predict模式，还需修改dir_origin_path、dir_save_path参数。
3.  可根据检测效果微调newmask_CenterNet文件中的confidence参数，建议从0.2开始调整。
###   使用CenterNet-GCA训练新的卵石数据集
1.  根据需要，修改newmask_train文件中的相关训练参数。
2.  微调训练时需设置pretrained参数为true，并将已有的权重写入model_path，推荐使用UD00作为初始权重。
##   检测示例
以下检测结果为使用权重UD00得到
![Alt Text](img_out/1.JPG)
![Alt Text](img_out/6.JPG)
##   注意事项
1.  仅对主干网络为ResNet50的情况进行了相应改进，代码中所有的backbone参数只支持resnet50。
2.  检测大尺寸图片时需要先裁剪为小尺寸图片（最好不超过1500*1000），再使用CenterNet-GCA识别。
##   参考
-  https://github.com/bubbliiiing/centernet-pytorch
-  https://github.com/xingyizhou/CenterNet
