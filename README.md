论文笔记
====

读论文要点：
----

写论文introduction就像写童话一样：<br>
1.有一条巨龙抓走了公主（介绍你的问题为什么值得研究）<br>
2.巨龙多么多么难打（强调你研究的重要性）<br>
3.王子提着一把金光闪闪的剑而不是破斧子烂长矛登场（你的方法好在哪里，别人的问题在哪里）<br>
4.王子是如何打败巨龙（你的方法简介）<br>
5.从此王子和公主幸福的生活在了一起（解决了问题）<br>

[GAN 论文笔记](./GAN%20note.md)

[RCNN（CVPR，2014）](https://cjmcv.github.io/deeplearning-paper-notes/fdetect/2016/01/02/RCNN.html)
----

需补充知识（Alexnet；VGG16）

作者：RBG（Ross B. Girshick）大神，不仅学术牛，工程也牛，代码健壮，文档详细，clone下来就能跑。
RCNN脉络

RCNN--SPPNET--Fast-RCNN--Faster-RCNN


![](https://pic2.zhimg.com/v2-0c98fb30a9e589fa164d99c50e6ca711_r.jpg)

参考资料:

[图解CNN论文：尝试用最少的数学读懂深度学习论文](https://www.bilibili.com/video/av22822657/)

[RCNN- 将CNN引入目标检测的开山之作](https://zhuanlan.zhihu.com/p/23006190)

RCNN (论文：Rich feature hierarchies for accurate object detection and semantic segmentation) 是将CNN方法引入目标检测领域， 大大提高了目标检测效果，可以说改变了目标检测领域的主要研究思路， 紧随其后的系列文章：（ RCNN）,Fast RCNN, Faster RCNN 代表该领域当前最高水准。

作者：周新一
链接：https://www.zhihu.com/question/35887527/answer/77490432
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

RCNN的一个出发点在文章第一句话：features matter。传统的物体检测使用hand-engineered的特征。如HOG特征可以表示为filter(convolutional)+gating(nonlinear)+pooling+normalization(LRN)，很多传统特征可以用图片经过CNN后得到的feature map表示。并且，由于卷积操作具有平移不变形，feature map里不仅包含了物体的what信息，还包含着物体的where信息。因此可以利用CNN特征来代替传统特征，对特征用SVM得到分类结果，同时可以对特征作回归得到更精确的位置。

SPP首先解决传统的CNN输入需要图片长宽固定的问题（这个问题在基于区域的输入上更为明显），把原来的“从原图上截取区域”转化为“从feature map上截取区域”，有了这个操作，同一张图片上不同区域的物体检测可以共享同一个原图特征。然而SPP的区域特征提取是在卷积层参数固定后才进行的，并且要额外进行SVM分类和区域框回归。

Fast RCNN解决了两个技术问题：1）如何让区域的选择可导（提出ROI Pooling操作，区域特征选择像Max Pooling一样可导），2）如何让SGD高效（把同一张图的不同区域放在同一个batch里），并且把原来的区域框回归作为一个Multitask接在网络里（Smooth L1 norm），这样除了区域推荐外的全部任务可以在一个网络完成。Faster RCNN把区域推荐也放在网络里完成（RPN）。这样整个框架都可以在一个网络里运行。

然而，Faster RCNN的训练为了保证卷积特征一致性需要分4步训练。RBG在Tutorial上表示已有让区域推荐可导（类似Spatial Transform Network中的sampler操作）的joint training。

论文算法概述

   整个算法过程包含三个步骤：a、输入图像，搜索生成类别独立的物体候选框； b、使用大的CNN网络对各候选框提取固定长度的特征向量；c、利用线性SVM对特征向量进行分类识别。
   
RCNN算法分为4个步骤 

候选区域生成： 一张图像生成1K~2K个候选区域 （采用Selective Search 方法）

特征提取： 对每个候选区域，使用深度卷积网络提取特征 （CNN） 

类别判断： 特征送入每一类的SVM 分类器，判别是否属于该类 

位置精修： 使用回归器精细修正候选框位置 (重叠度（IOU）+非极大值抑制（NMS）)

   
  ![](https://cjmcv.github.io/deeplearning-paper-notes/images/pdDetect/rcnn1.jpg)
  
存在问题
  
训练过程繁琐：包含基于物体候选框使用log损失微调卷积网络；基于调优后的卷积特征训练SVM；训练bounding box回归器（参考DPM）；
训练耗时，占用磁盘空间大。
每个候选框需要进行整个前向CNN计算导致测试速度慢：

总结

算法分三步，1）selective；2）将各候选框缩放填充至固定大小，用一个CNN对分别提取特定长度的特征向量；3）SVM对向量分类。主要短板在第2步，每个框都要重复输入CNN进行独立的计算。
  
  [SPP-Net（ECCV，2014）](https://cjmcv.github.io/deeplearning-paper-notes/fdetect/2016/02/05/SPP.html)
  ----
  
  
  论文算法概述
  
   SPP（spatial pyramid pooling，空间金字塔池化）：1）SPP层可以不论输入大小都能产生固定长度的输出，而传统基于滑动窗口的池化层则不行；2）SPP层使用多空间尺度的bins，而基于滑动窗口的池化层则使用单一尺寸窗口。多尺度的池化对于物体变型更具有鲁棒性；   3）由于可以输入任意尺度，SPP可以提取多尺度的池化特征；
  
 1、 R-CNN中输入的候选窗口需要归一化到一定大小，如图所示裁剪和变换都可能会造成信息丢失导致识别率下降。而SPP-net可以解决这个问题。

2、R-CNN在检测图像时使用selective search提取出很多个候选框（2000个），然后每个候选框都各自进行一次前向CNN特征提取与线性SVM分类，因此很耗时。而SPP-net则避免了重复计算卷积特征。

何凯明团队的SPPNet给出的解决方案是，既然只有全连接层需要固定的输入，那么我们在全连接层前加入一个网络层，让他对任意的输入产生固定的输出不就好了吗？一种常见的想法是对于最后一层卷积层的输出pooling一下，但是这个pooling窗口的尺寸及步伐设置为相对值，也就是输出尺寸的一个比例值，这样对于任意输入经过这层后都能得到一个固定的输出。SPPnet在这个想法上继续加入SPM的思路，SPM其实在传统的机器学习特征提取中很常用，主要思路就是对于一副图像分成若干尺度的一些块，比如一幅图像分成1份，4份，8份等。然后对于每一块提取特征然后融合在一起，这样就可以兼容多个尺度的特征啦。SPPNet首次将这种思想应用在CNN中，对于卷积层特征我们也先给他分成不同的尺寸，然后每个尺寸提取一个固定维度的特征，最后拼接这些特征不就是一个固定维度的输入了吗？
![](https://cjmcv.github.io/deeplearning-paper-notes/images/pdDetect/spp1.jpg)
![](https://cjmcv.github.io/deeplearning-paper-notes/images/pdDetect/spp2.jpg)

存在问题

1、和RCNN一样，训练过程繁琐，包含基于物体候选框使用log损失微调卷积网络；基于调优后的卷积特征训练SVM；训练bounding box回归器（参考DPM）；

2、和RCNN一样，训练结果占用空间很大，需要保存在磁盘中；

3、与RCNN不同，SPP-Net无法更新SPP层前面的卷积层，因为CNN调优和SVM训练分离，SVM的loss无法传播到卷积层（RCNN也一样）,即意味着只能微调全连接层，而对于很多情况下卷积层也是需要微调的，特别是层次更深的网络。

实验结果

其中单一尺寸训练结果低于RCNN1.2%，但是速度是其102倍，5个尺寸的训练结果与RCNN相当，其速度为RCNN的38倍。

总结

SPP叫空间金字塔池化，主要是为了处理RCNN中需要将候选框的大小归一化造成的损失，以及每个框单独重复计算的问题。
这里前面的操作与RCNN一致，不同点是对输入全图图像做一次CNN前向，然后在根据SS的候选框去切分特征图，得到一个个大小不定的特征图块，将每个特征图块输入到一个SPP层上得到一个固定长度的特征向量，再进入fc层。

即把候选框切分图像块改成切分特征图块，用SPP层解决固定长度的问题。

PS:

空间金字塔池化，设金字塔有三层，分别对应1x1、2x2和4x4的池化，设输入特征图通道数为256，则1x1池化得到1x1x256d向量，2x2池化得到2x2x256d，4x4池化得到4x4x256d，则最终的特征向量长度为三个数累加，维度固定，与输入特征图大小无关。

[Fast-RCNN（ICCV，2015）](https://cjmcv.github.io/deeplearning-paper-notes/fdetect/2016/01/04/FRCNN.html)
----
参考资料

[Fast Rcnn git hub 代码](https://github.com/yhenon/keras-frcnn)

[图解Fast RCNN](https://www.bilibili.com/video/av22822657/?p=44)

回顾RCNN

在RCNN中，可以看到还是存在很多问题。其中最主要的就是训练过程复杂：

先预训练

再利用selective search产生的region proposals进行微调CNN网络

再用CNN提取到的region proposal的固定长度的特征向量来训练一组SVM

最后还要训练一组BBox回归器，还是需要用CNN来提取特征

其中2，3，4中样本数据的构造是不一样的，所以还要分别构造各自的数据集。训练过程中反复用到CNN（要么就用额外的存储来保存特征向量，代价很大）。

除了训练，测试过程也比较慢，因为需要将产生的2k个region proposals依次通过CNN来提取特征（没有共享计算导致的结果）。

当然这些都不能否定RCNN在图像检测方面的突破。下面开始进入Fast R-CNN.

改进

很大程度上实现了end to end（除了region proposals的产生还是用的selective search）。

不再是将region proposals依次通过CNN，而是直接输入原图，来提取特征（这样一张图只会CNN一次）。

网络同时输出类别判断以及bbox回归建议（即两个同级输出），不再分开训练SVM和回归器。

![网络结构](https://pic4.zhimg.com/80/v2-d825cbb7a7ab0d15c559c2d595b6a4a2_hd.jpg)

![具体网络结构](https://pic3.zhimg.com/80/v2-1ef48d2473e783a4bdc7d66ee5f8a083_hd.jpg)

[FASTER RCNN](https://cjmcv.github.io/deeplearning-paper-notes/fdetect/2016/01/06/FRCNN2.html)
----
参考资料

[keras版faster-rcnn算法详解](https://zhuanlan.zhihu.com/p/28585873)

[faster -rcnn 代码详解](https://zhuanlan.zhihu.com/p/31530023)

[目标检测——从RCNN到Faster RCNN 串烧](https://blog.csdn.net/xyy19920105/article/details/50817725)

经过R-CNN和Fast RCNN的积淀，Ross B. Girshick在2016年提出了新的Faster R-CNN，在结构上，Faster RCNN已经将特征抽取(feature extraction)，proposal提取，bounding box regression(rect refine)，classification都整合在了一个网络中，使得综合性能有较大提高，在检测速度方面尤为明显。

![](https://pic1.zhimg.com/80/v2-e64a99b38f411c337f538eb5f093bdf3_hd.jpg)

![Faster RCNN训练步骤](https://pic1.zhimg.com/80/v2-ed3148b3b8bc3fbfc433c7af31fe67d5_hd.jpg)

总结
主要是提出了RPN，代替了Fast-RCNN上的使用的selective-search，其余结构未变。而RPN中设定3个尺度和3个形状组成9个anchors，以滑动窗口的方式搜索目标。其中滑动窗口以n x n卷积实现，卷积输出多个特征图，即使每个滑动窗口得到一个多维向量。然后再用1x1x18和1x1x36卷积去扫描，分别得到对应每个位置上9个anchor的类别和边框的预测。

[A-Fast-RCNN（CVPR, 2017）](https://cjmcv.github.io/deeplearning-paper-notes/fdetect/fgan/2017/04/30/AFastRCNN.html)
----

Motivation

这篇文章提出了一种新的对手生成策略，通过训练提升检测网络对遮挡、形变物体的识别精度。

遮挡和形变是检测任务中影响模型性能的两个显著因素。增加网络对遮挡和形变的鲁棒性的一个方式是增加数据库的体量。但是由于遮挡的图片一般都处在图像分布的长尾部分，即便增加数据，遮挡和形变的图片仍是占比较少的部分。另一个思路就是使用生成网络产生遮挡和形变图片。然而遮挡和形变的情况太多，直接生成这些图片还是比较困难的事情。在这篇文章中，作者的思路是训练一个对手网络，让这个网络动态产生困难的遮挡和形变样本。以遮挡为例，作者希望被训练的检测网络会认为哪里的遮挡更难以处理。之后去遮挡这些区域的特征，从而让检测网络努力学习这些困难的遮挡情况。对于形变的情况同理。与其它提升检测网络性能的方法，如换用更强大的主干网络、使用自上而下的网络设计等相比，本文提出的方法则是考虑如何更加充分地利用训练数据。

[作者：kfxw
链接：https://zhuanlan.zhihu.com/p/33936283
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。]

![](https://cjmcv.github.io/deeplearning-paper-notes/images/pdDetect/afastrcnn1.png)

Experiments

在VOC2007上的结果如下图所示，与Online Hard Example Mining（OHEM）相比，在VOC2007上文中结果（71.4%）比OHEM（69.9%）要好，但在VOC2012上文中结果（69.0%）比OHEM（69.8%）要稍低一点。作者认为这两种方法可以互补，实验中将两种方法联合，在VOC2012上可以达到71.7%；而两个OHEM模型联合为71.2%，两个文中的模型联合则为70.2%。

[VGG（ICLR, 2014）](https://cjmcv.github.io/deeplearning-paper-notes/freg/2016/02/18/VGG.html)
----
googlenet和vggnet这两个模型是在AlexNet后人工神经网络方面研究的又一里程碑，也是许多论文和博客中用来和提出的新方法进行对比的baseline。理解这两个网络并且明白它们的优劣对走在机器学习之路上的小伙伴们来说是非常重要的。这两类模型结构有一个共同特点是Go deeper，但是在具体实现上却有很大差异。

作者：人工智能LeadAI
链接：https://zhuanlan.zhihu.com/p/33454226
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
论文概述

   论文主要探索了卷积网络的深度对识别准确率的影响，构建了多个网络进行分析。
   
   
  ![](https://cjmcv.github.io/deeplearning-paper-notes/images/pdReg/vgg1.png)

单尺度

注意到在没有任何归一化层的模型A中使用local response normalisation (A-LRN network)也并不能起到提升作用，因此也没有在更深的网络（B-E）中使用归一化层。

注意到随着卷积层数增加分类误差减少（11层到19层），在这些网络中，层数到19层后错误率达到饱和，但更深的层数可能对更大的数据集有帮助。

尽管网络深度一致，但C网络（包含三个11卷积层）比D网络（整个网络都使用33的卷积层）效果差，这意味着尽管额外的非线性层能起到辅助作用，但使用不同感受野的卷积提取空间信息也很重要；一个深的网络使用小的卷积滤波器比浅的网络使用大的卷积滤波器效果要好。

训练图像尺度的微小变化[256,512]比使用固定大小的效果好不少，尽管在测试时都是使用固定尺度。这证明使用尺度缩放进行训练数据扩增对于多尺度图像检测有一定作用。

[googlenet(2014)](https://cjmcv.github.io/deeplearning-paper-notes/freg/2016/02/16/GoogLeNet.html)
----

论文算法概述

   提出一种深度卷积神经网络框架，叫做Inception。一个典型的应用例子是22层的GoogLeNet，取得ILSVRC14中物体分类与检测冠军。这种框架基于Hebbian规则和多尺度处理的思想进行设计，其主要特点是可以在固定的运算量下提升网络的深度和宽度，使能更好利用网络中的计算资源。且GoogLeNet的参数比2012年的Alexnet少12倍，准确率却高不少。主要采用了1×1卷积的方式，主要用于实现两个功能：1）作为降维模块去除限制网络大小的计算瓶颈;  2）使可以加深加宽网络结构而不降低网络性能。
   
   ![google net 网络模型](https://pic3.zhimg.com/v2-54aeea686f45367dcb82934c19f6de4e_b.jpg)
   

