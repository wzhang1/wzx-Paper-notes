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
 
 Global Upsampling Network (GN)
 
GN网络有两条支路并行处理，在图像上采样的支路中使用反卷积层得到一个大的平滑而缺少细节的图像，使用双线性插值矩阵去初始化反卷积层权重，允许这些权重随着训练而更新，尽管反卷积层权重会更新，但不会更新太多使输出的是正常的平滑上采样图。全局细节生成的支路由全连接层作为编码层的网络实现，在除了最后用于生成128x128的上采样全局细节的层外，每一层的特征图都接ReLU。而且这里编码层无论是上采样4倍还是8倍都采用256维，这主要是因为训练样本有限，全局特征训练容易出现过拟合。最后将上采样的网络支路和全局细节特征生成支路的输出进行拼接，得到2x128x128的张量用于LN。


Local Refinement Network (LN)


LN的结构如图所示，分别对应上采样4倍和8倍任务，分析了三个有不同层数的全卷积网络框架。在每次卷积操作时都对图像做padding保持大小一致，卷积后输入到ReLU。全称没有使用池化，并且滑动步长为1，因此网络学习到了平移不变的非线性，如图2(c)，LN通过由GN得到的平滑和细节层，加强了人脸特定的局部特征。此外，重构图的局部数据分布与高分辨率对应图像块的数据分布相吻合（例如平滑的脸颊区域和尖锐的脸部轮廓）。

[SRGAN](https://cjmcv.github.io/deeplearning-paper-notes/fgan/fsr/2016/12/26/SRGAN.html)
----

论文概述：

本论文中，使用生成对抗网络GAN用于超分辨率重构。提出了一种感知损失（perceptual loss），包含有对抗损失和内容损失（content loss）。对抗损失基于鉴别器使生成图像更接近真实图像；内容损失由直观相似性（perceptual similarity）驱动而不是像素域上的相似性。文中以深度残差网络作为生成模型，以VGG网络作为鉴别器，可以从高度下采样的图像中恢复图像真实的纹理信息。大量的mean-opinion-score（MOS，平均主观意见分）测试表明使用SRGAN在视觉质量（perceptual quality）上有巨大意义。

  对于大的上采样因子，SR通常存在纹理细节缺失的问题，如图中SRResNet的PSNR评分很高，但从服饰/叶子等可以看出，也存在细节缺失的问题。而文中提出的算法SRGAN虽然PSNR分数没有SRResNet的高，但可生成很多细节信息，使在人类视觉感官上图像质量更高。注：论中算法评价标准为MOS，而不是PSNR或SSIM。
  
  训练细节与实验结果
  
从ImageNet中随机抽取35w样本，通过双立方插值进行4倍下采样。从不同的16张图像上随机裁剪96x96的图像块作为一个mini-batch。基于MSE优化的SRResNet作为GAN的生成器初始化预训练模型以避免陷入局部最优，交替训练。在测试的时候将batch-normalization的更新关掉，使输出仅取决与输入。
