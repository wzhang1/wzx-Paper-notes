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

[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf )
----
Abstract

 提出通过一个对抗过程来评估生成模型，在对抗的过程中同时训练两个网络，一个是生成模型G用于获取数据的分布情况，另一个是区分模型D用于判断一个样本是来自训练集还是由模型G生成。训练中D目标是最大化地将原始训练样本和由G生成的样本准确区分，即二分类问题，最大化log D(x)；G则要最大化模型D出现评估错误的概率log(1 – D(G(z)))，而GAN模型没有损失函数，优化过程是一个“二元极小极大博弈（minimax two-player game）”问题。则最后达到最优解时，G可以再现训练样本的分布，而D的判断准确率为50%。
 
Advantages and disadvantages：

优：

计算量小，不需要使用马尔科夫链反复采样，也不用在学习中推论（如wake-sleep），只需要使用BP来计算梯度，并且大量的函数可以被包含到模型中。

具有一些统计上的优势，因为生成网络的参数是由样本通过梯度传播来训练的，而并不是由样本直接拷贝而来。

对抗网络生成的样本更锐利清晰，甚至对于退化的分布也一样。而基于马尔科夫链的方法需要提供有点模糊的分布以便链能够混合各种分辨率？

缺： 由模型G得到的数据分布没有一个明确的表示，没有损失函数，在过于自由不可控,训练过程中很难区分是否在向好的方向训练。而且在训练时模型D需要和模型G同步得很好，否则可能发生崩溃问题，生成器开始退化，总是生成同样的样本点，无法继续学习。当生成模型崩溃时，判别模型也会对相似的样本点指向相似的方向，训练无法继续。训练较困难。

结论与展望

一个条件生成模型p(x	c)可以由将c作为输入添加到模型G和D中得到。
学习近似推理可以通过训练辅助网络来基于x预测z，这是类似于用wake-sleep算法训练的推理网络，但该推理网络具有的优势是在训练完毕后能一个固定的生成器。

半监督学习：来自判别模型D或推论网络的特征在带标签数据有限的情况下有助于提升分类器的性能。

效率提升：通过更好的协调G和D或者在训练中采用更好的分布去采样z能够对训练起到加速作用。

[GLN（2016）](https://cjmcv.github.io/deeplearning-paper-notes/fsr/fgan/2016/12/15/GLN.html)
----


 人脸幻构任务在输入图像分辨率非常小(如10x12像素等)或者在不可控的多姿态多亮度情况下具有很大的挑战性。该论文在07年的Face Hallucination: Theory and Practice中的框架上做改进，提高了精度和效率，利用全局和局部约束使人脸可以高效地被模型化，并且该网络可以进行端到端训练。从理论上，该网络可以分为两个子网络，一个根据全局约束实现了整体人脸的重构，另一个则强化了人脸特定的细节部分并约束了局部图像块的数据分布统计。使用了一个用于超分辨率重构的新损失函数，结合了与训练人脸质量的重构误差作为对抗，使输出有更好的视觉效果。实验证明该方法在数据上和视觉上都达到了最先进水平。
 
 ![](https://cjmcv.github.io/deeplearning-paper-notes/images/pdSr/gln1.png)

Global Upsampling Network (GN)
 
GN网络有两条支路并行处理，在图像上采样的支路中使用反卷积层得到一个大的平滑而缺少细节的图像，使用双线性插值矩阵去初始化反卷积层权重，允许这些权重随着训练而更新，尽管反卷积层权重会更新，但不会更新太多使输出的是正常的平滑上采样图。全局细节生成的支路由全连接层作为编码层的网络实现，在除了最后用于生成128x128的上采样全局细节的层外，每一层的特征图都接ReLU。而且这里编码层无论是上采样4倍还是8倍都采用256维，这主要是因为训练样本有限，全局特征训练容易出现过拟合。最后将上采样的网络支路和全局细节特征生成支路的输出进行拼接，得到2x128x128的张量用于LN。


Local Refinement Network (LN)


LN的结构如图所示，分别对应上采样4倍和8倍任务，分析了三个有不同层数的全卷积网络框架。在每次卷积操作时都对图像做padding保持大小一致，卷积后输入到ReLU。全称没有使用池化，并且滑动步长为1，因此网络学习到了平移不变的非线性，如图2(c)，LN通过由GN得到的平滑和细节层，加强了人脸特定的局部特征。此外，重构图的局部数据分布与高分辨率对应图像块的数据分布相吻合（例如平滑的脸颊区域和尖锐的脸部轮廓）。

[SRGAN](https://cjmcv.github.io/deeplearning-paper-notes/fgan/fsr/2016/12/26/SRGAN.html)
----

论文概述：

本论文中，使用生成对抗网络GAN用于超分辨率重构。提出了一种感知损失（perceptual loss），包含有对抗损失和内容损失（content loss）。对抗损失基于鉴别器使生成图像更接近真实图像；内容损失由直观相似性（perceptual similarity）驱动而不是像素域上的相似性。文中以深度残差网络作为生成模型，以VGG网络作为鉴别器，可以从高度下采样的图像中恢复图像真实的纹理信息。大量的mean-opinion-score（MOS，平均主观意见分）测试表明使用SRGAN在视觉质量（perceptual quality）上有巨大意义。

![对比结果](https://cjmcv.github.io/deeplearning-paper-notes/images/pdGan/srgan1.png)

  对于大的上采样因子，SR通常存在纹理细节缺失的问题，如图中SRResNet的PSNR评分很高，但从服饰/叶子等可以看出，也存在细节缺失的问题。而文中提出的算法SRGAN虽然PSNR分数没有SRResNet的高，但可生成很多细节信息，使在人类视觉感官上图像质量更高。注：论中算法评价标准为MOS，而不是PSNR或SSIM。
  
  训练细节与实验结果
  
从ImageNet中随机抽取35w样本，通过双立方插值进行4倍下采样。从不同的16张图像上随机裁剪96x96的图像块作为一个mini-batch。基于MSE优化的SRResNet作为GAN的生成器初始化预训练模型以避免陷入局部最优，交替训练。在测试的时候将batch-normalization的更新关掉，使输出仅取决与输入。

[SimGAN](https://cjmcv.github.io/deeplearning-paper-notes/fgan/2017/01/08/SimGAN.html)
----

论文算法概述
   人工合成的图像和真实图像分布存在一定差异，所以直接从人工合成样本中进行训练难以达到预期的效果。为减少这个差异，论文中提出Simulated+Unsupervised (S+U) learning“模拟+无监督”学习方法，在保留从网络输出的注释信息的同时使用无标签的真实数据使simulator生成数据更接近现实。这种方法与GANs生成对抗网络相似，但这里是以合成图像作为输入，而不是GAN中的以随机向量作为输入。相对于标准的GAN，有几点关键的修改：自正则化(self-regularization)，局部对抗性损失和使用精炼图像refined images去优化鉴别器。该方法在没有使用任何标注数据的情况下在MPIIGaze数据库上达到最高水平的效果。

![](https://cjmcv.github.io/deeplearning-paper-notes/images/pdGan/simgan1.png)

 上图为SimGAN的概要图，使用一个refiner网络R来对由模拟器simulator生成的图像进行精炼(refine)，使最小化局部对抗损失(local adversarial loss)和自正则化(self regularization)。其中对抗损失迷惑用于判断图像是真实图像还是精炼图像的鉴别器D，而自正则化用于最小化生成图像和精炼图像的差异。这样保留了注释信息（如图中的视线方向），使精炼得到的图像适合于模型训练。在训练时，精炼网络R和鉴别器网络D是交替更新的。
 
 模拟+无监督学习
 
S+U学习的目的是使用无标签真实图像集去训练精炼网络R去提炼由模拟器网络得到的图像，在保留生成网络中该图像的注释信息的同时，使精炼图像看起来更像真实图像。精炼网络R的参数theta，由两个loss进行监督训练，其中xi为训练样本，xi~为相应的精炼图像，第一部分是在合成图像中增加真实性的成本，第二部分是通过最小化合成图像与精炼图像的差异来保留注释信息的成本。

根据精炼图像的历史情况去更新鉴别器

对抗训练的另一个问题是鉴别器只关注最后的精炼图像结果进行训练，这样会导致有两个问题，一个是分散了对抗训练，二是精炼器会再次引入鉴别器曾经关注过而当前没关注的人工合成信息。在训练过程中的任何时刻从精炼器中得到的精炼图像，对于鉴别器来说都属于‘假’的一类，所以鉴别器应可以把这些图像都分到‘假’一类，而不仅只针对当前生成的mini-batch个精炼图像。通过简单修改Algorithm1，使采用精炼器的历史情况来更新鉴别器。令B为由以往精炼器生成的精炼图像集的缓存，b为mini-batch大小，在鉴别器训练的每次迭代中，通过从当前精炼器中采样b/2的图像来计算鉴别器的损失函数，然后从缓存中采样额外的b/2的图像来更新参数。保持缓存B大小固定，然后在每次迭代中随机使用新生成的精炼图像去替换缓存中b/2个图像，如图4。
![](https://cjmcv.github.io/deeplearning-paper-notes/images/pdGan/simgan6.png)

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

