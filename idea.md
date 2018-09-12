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

根据道路蒙板生成道路图像

或者用激光数据和图像数据共同训练神经网络，，检测道路

神经网络检测道路遮挡

mobilenet-ssd 换成 mobilenet yolo 并对mobilenet进行优化 （https://www.jianshu.com/p/681960b4173d?from=groupmessage）


mobilenet yolo (https://github.com/zunzhumu/darknet-mobilenet)这个是权重把它替换进https://github.com/qqwweee/keras-yolo3 里面的配置文件

SSGAN:半监督对抗神经网络
将真实数据与生成数据拼接后扔去鉴别器

[](https://github.com/XinyuGong/SSGAN-Tensorflow)

![](https://github.com/XinyuGong/SSGAN-Tensorflow/raw/master/figure/ssgan.png)
