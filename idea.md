1.可变型卷积核(效果更好，计算量更小)

项目地址：https://github.com/msracver/Deformable-ConvNets

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