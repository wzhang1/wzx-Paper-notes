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

[RCNN 系列](./RCNN系列.md)


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
   

