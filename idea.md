1.可变型卷积核(效果更好，计算量更小)

项目地址：https://github.com/msracver/Deformable-ConvNets

新闻地址附使用方法:http://www.sohu.com/a/138675188_465975

论文地址：https://arxiv.org/abs/1703.06211

2.Mobilenet V2（体积小，性能好,使用了DW卷积）

3.二值化神经网络(减少存储空间)

4.SEnet (那为什么要认为所有通道的特征对模型的作用就是相等的呢,该网络使用了对每个通道加入了权重并训练)

我们可以看到这样一些趋势：

卷积核方面：

大卷积核用多个小卷积核代替；
单一尺寸卷积核用多尺寸卷积核代替；
固定形状卷积核趋于使用可变形卷积核；
使用1×1卷积核（bottleneck结构）。

卷积层通道方面：

标准卷积用depthwise卷积代替；
使用分组卷积；
分组卷积前使用channel shuffle；
通道加权计算。
卷积层连接方面：

使用skip connection，让模型更深；
densely connection，使每一层都融合上其它层的特征输出（DenseNet）
启发

类比到通道加权操作，卷积层跨层连接能否也进行加权处理？bottleneck + Group conv + channel shuffle + depthwise的结合会不会成为以后降低参数量的标准配置？


SCNN :特殊卷积核，有助于提升空间信息

使用DCGAN将道路图变得更加完善，把遮挡什么的都去掉，但是用处没有想到
----

数据预处理，是灰度图像更加清晰，增强鲁棒性

根据消失点预估车道线位置

使用DCGAN补全



根据道路蒙板生成道路图像

或者用激光数据和图像数据共同训练神经网络，，检测道路

神经网络检测道路遮挡，车牌检测，车牌遮挡检测

mobilenet-ssd 换成 mobilenet yolo 并对mobilenet进行优化 （https://www.jianshu.com/p/681960b4173d?from=groupmessage）


mobilenet yolo (https://github.com/zunzhumu/darknet-mobilenet)这个是权重把它替换进https://github.com/qqwweee/keras-yolo3 里面的配置文件

SSGAN:半监督对抗神经网络
将真实数据与生成数据拼接后扔去鉴别器

[](https://github.com/XinyuGong/SSGAN-Tensorflow)

![](https://github.com/XinyuGong/SSGAN-Tensorflow/raw/master/figure/ssgan.png)





ETH Zurich提出利用对抗策略，解决目标检测的域适配问题
----

![](http://static.extremevision.com.cn/donkey_df1e099a-f9cd-4dc7-895a-823e2a527863.jpg)


背景介绍

基于有监督的目标检测器（如Faster R-CNN, SSD）在有标签数据集上（Pscal VOC, COCO等）取得了很好的性能。然而，一方面检测器的训练严重依赖大量有标签数据，在现实场景中这些有标签数据的获取代价是很大的。另一方面，在一个场景上训练的模型难以generalize到另一个场景中，例如在自动驾驶场景下，当遇到没有标签数据的新场景时，如何使得旧场景上训好的模型在新场景上也有很好的表现？

域适配（Domain Adaptation, DA）问题已经在图像分类任务上得到了广泛研究并取得了惊人进展，今年CVPR上也有很多相关工作。其本质属于迁移学习的一种，问题设定是：如何使得源域（Source Domain）上训练好的分类器能够很好地迁移到没有标签数据的目标域上（Target Domain）上。

其中两个代表性的工作有：DSN [1]和ADDA [2]。

本文的两点贡献：

提出图像层面的适配（Image-Level Adaptation）和目标层面的适配（Instance-Level Adaptation），

用于解决自动驾驶场景下目标检测任务使用不同数据训练的域适配问题。

[代码](https://github.com/yuhuayc/da-faster-rcnn)


CoGAN:

[code](https://github.com/andrewliao11/CoGAN-tensorflow)

文章的思想是,利用网络层的权重共享约束,训练GAN网络.模型包括两个生成网络,两个判别网络,

![](https://github.com/andrewliao11/CoGAN-tensorflow/raw/master/asset/network.png?raw=true)

ADDA:

Adversarial Discriminative Domain Adaption 

文章利用GAN网络的思想用于cross-domain识别

使用一个domain二分类器（简单的全连接神经网络）将获取的特征进行分类，然后定义一个domain confusing loss，通过优化特征提取让该domain二分类器分辨不出他们来。

![](https://github.com/greenfishflying/wzx-Paper-notes/blob/master/image/ADDA1.png)

CVPR 2018 | ETH Zurich提出新型网络「ROAD-Net」，解决语义分割域适配问题

本文研究的是无人驾驶场景中的语义分割问题。语义分割的样本标记成本很高，使用合成数据能帮助解决样本不足问题。但是合成的数据和真实的数据之间存在差异，这种差异会极大影响使用合成数据训练的模型在真实数据上的表现。

本文研究难点在于如何处理合成数据和真实数据之间的差异

![](http://static.extremevision.com.cn/donkey_fd09081e-6d4e-47fc-a16c-8b3f69031f5b.jpg)


![](http://static.extremevision.com.cn/donkey_9d76a7a3-884e-47e0-a0a7-8c8c8a74f74f.jpg)
