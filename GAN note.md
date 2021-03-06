
[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf )
----

![各种GAN的结构图](https://github.com/greenfishflying/tensorflow-generative-model-collections/raw/master/assets/etc/GAN_structure.png)

[千奇百怪的GAN变体，都在这里了](https://zhuanlan.zhihu.com/p/26491601)

[GAN 进展跟踪 10 大论文，不容错过（附下载）](https://zhuanlan.zhihu.com/p/34132477)

[pytorch-CycleGAN-and-pix2pix代码](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

[GAN论文汇总（含代码）](https://github.com/zhangqianhui/AdversarialNetsPapers)

[各种GAN结构（笔记）](https://blog.csdn.net/u012969412/article/details/79135848)

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



cGAN:
----


为了提高训练的稳定性，另一个很自然的角度就是改变学习方法。把纯无监督的 GAN 变成半监督或者有监督的。这便可以为 GAN 的训练加上一点点束缚，或者说加上一点点目标.中提出的 Conditional Generative Adversarial Nets （CGAN）便是十分直接的模型改变，在生成模型（G）和判别模型（D）的建模中均引入 conditional variable y，这个 y 就是数据的一种 label。也因此，CGAN 可以看做把无监督的 GAN 变成有监督的模型的一种改进。这个简单直接的改进被证明非常有效，并广泛用于后续的相关工作中。

[cGANs with Projection Discriminator](https://link.zhihu.com/?target=https%3A//openreview.net/pdf%3Fid%3DByS1VpgRZ)
----

[代码](https://github.com/pfnet-research/sngan_projection)

这篇论文提出了一种新的、基于投影的方法，将有条件的信息（conditional information）纳入 GAN 的判别器。这种方法与当前的大多数条件 GAN（cGAN）的框架不同，它是通过将（嵌入的）条件向量连接到特征向量来使用条件信息。通过这样的修改，研究者在 ImageNet 的 class conditional 图像生成质量比当前最优结果显著提高，并且这是只通过一对 discriminator 和 generator 实现的。该研究还将应用扩展到超分辨率，并成功地生成了高质量的超分辨率图像。代码、生成的图像和预训练的模型可用。

![](https://pic4.zhimg.com/80/v2-0a258e964f0e06227e8d294d99e7c6bd_hd.jpg)

[High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://tcwang0509.github.io/pix2pixHD/)
----

研究者提出一种多尺度的生成器和判别器架构，结合新的对抗学习目标函数。实验结果表明，条件 GAN 能够合成高分辨率、照片级逼真的图像，不需要任何手工损失或预训练的网络。

不仅如此，作者还提出了一种方法，让用户能够交互式地编辑物体的外观，大大丰富了生成的数据类型。例如，在下面的视频中，你可以发现用户能够选择更换街景中车辆的颜色和型号，给街景图增加一些树木，或者改变街道类型（例如将水泥路变成十字路）。类似地，利用语义标注图合成人脸时，给定语义标注的人脸图像，你可以选择组合人的五官，调整大小肤色，添加胡子等。

作者在文中指出，他们的方法可以扩展到其他领域，尤其是医疗图像这样缺乏预训练网络的领域。

![](https://pic1.zhimg.com/80/v2-0b181283559f38a23c2f9f8604918c34_hd.jpg)

[ StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1710.10916.pdf)
----

尽管生成的敌对网络 (GANs) 在各种任务中已经取得了显著的成功，但它们仍然在生成高质量图像方面面临挑战。本文提出了一种堆叠的生成对抗网络(StackGAN)，目标是生成高分辨率的现实图像。

首先，本文提出了一个包含两阶段的生成对抗网络架构 stack GAN-v1 用于文本 - 图像合成。根据给定的文字描述，GAN 在第一阶段描绘出了物体的原始形状和颜色，产生了低分辨率的图像。在第二阶段，GAN 将第一阶段的低分辨率图像和文字描述作为输入，并以逼真的细节生成高分辨率的图像。

其次，提出了一种多阶段的生成对抗性网络架构，即 StackGAN-v2，用于有条件和无条件的生成任务。提出的 StackGAN-v2 由多个树状结构的生成器和判别器组成。树的不同分支可以生成对应于同一场景的多个尺寸的图像。通过对多个分布的联合逼近，StackGAN-v2 显示了比 StackGAN -v1 更稳定的训练结果。大量的实验证明，在生成高清图像时，文章提出的堆叠的生成对抗网络比其他现阶段表现优异的算法更具优势。文章中提出的模型如图 1 所示：

![](https://pic2.zhimg.com/80/v2-501aaae4ef82ee9fb32631ff0cd7f4d3_hd.jpg)

[PG-GAN](https://openreview.net/pdf?id=ByS1VpgRZ)
----

这篇论文提出了一种新的、基于投影的方法，将有条件的信息（conditional information）纳入 GAN 的判别器。这种方法与当前的大多数条件 GAN（cGAN）的框架不同，它是通过将（嵌入的）条件向量连接到特征向量来使用条件信息。通过这样的修改，研究者在 ImageNet 的 class conditional 图像生成质量比当前最优结果显著提高，并且这是只通过一对 discriminator 和 generator 实现的。该研究还将应用扩展到超分辨率，并成功地生成了高质量的超分辨率图像。代码、生成的图像和预训练的模型可用。

[代码](https://github.com/tkarras/progressive_growing_of_gans)

来自 NVIDIA Research 的 GAN 论文，提出以一种渐进增大（progressive growing）的方式训练 GAN，通过使用逐渐增大的 GAN 网络（称为 PG-GAN）和精心处理的 CelebA-HQ 数据集，实现了效果令人惊叹的生成图像。作者表示，这种方式不仅稳定了训练，GAN 生成的图像也是迄今为止质量最好的。

它的关键想法是渐进地增大生成器和鉴别器：从低分辨率开始，随着训练的进展，添加新的层对越来越精细的细节进行建模。“Progressive Growing” 指的是先训练 4x4 的网络，然后训练 8x8，不断增大，最终达到 1024x1024。这既加快了训练速度，又大大稳定了训练速度，并且生成的图像质量非常高，例如 1024×1024 的 CelebA 图像。

![](https://pic3.zhimg.com/80/v2-54cfed8187b1cf5a02c086eae17670c2_hd.jpg)


![](https://pic3.zhimg.com/80/3b6c35178f34a004368770cda3ea41cd_hd.jpg)

 LAPGAN:
 ----
 
 将 GAN 的学习过程变成了 sequential “序列式” 的。具体上，LAPGAN 采用了 Laplacian Pyramid 实现了 “序列化” ，也因此起名做 LAPGAN 。
 
 [SN-GAN](https://openreview.net/pdf?id=B1QRgziT-)
 ----
 
 Goodfellow表示，虽然GAN十分擅长于生成逼真的图像，但仅仅限于单一类型，比如一种专门生成人脸的GAN，或者一种专门生成建筑物的GAN，要用一个GAN生成ImageNet全部1000种类的图像是不可能的。但是，这篇ICLR论文做到了。
 

摘要：

生成对抗网络的研究面临的挑战之一是其训练的不稳定性。在本文中，我们提出了一种叫做“谱归一化”（spectral normalization）的新的权重归一化（weight normalization）技术，来稳定判别器的训练。这种新归一化技术计算轻巧，易于并入现有的部署当中。我们在CIFAR10，STL-10和ILSVRC2012数据集上测试了谱归一化的功效，通过实验证实了相对于那些使用此前提出的训练稳定技术训练的GAN，谱归一化GAN（SN-GAN）能够生成质量相同乃至更好的图像。
 
 简单说，论文提出了一种新的权重归一化方法，用于稳定判别器的训练。作者在论文中写道，他们的归一化方法需要调整的超参数只要一个，就是 Lipschitz 常数，而且即使不调整这个超参数，也能获得满意的性能。此外，算法实现简单，额外的计算成本很小。
 
 ![](https://pic3.zhimg.com/80/v2-96087d2c2bf93447c3760c1348bdd728_hd.jpg)
 
 SN-GAN是所有方法中唯一训练成功了的，据我们所知，这也是首次用单对判别器和生成器从ImageNet数据集生成不错图像的尝试”。
 

[GLN（2016）](https://cjmcv.github.io/deeplearning-paper-notes/fsr/fgan/2016/12/15/GLN.html)
----


 人脸幻构任务在输入图像分辨率非常小(如10x12像素等)或者在不可控的多姿态多亮度情况下具有很大的挑战性。该论文在07年的Face Hallucination: Theory and Practice中的框架上做改进，提高了精度和效率，利用全局和局部约束使人脸可以高效地被模型化，并且该网络可以进行端到端训练。从理论上，该网络可以分为两个子网络，一个根据全局约束实现了整体人脸的重构，另一个则强化了人脸特定的细节部分并约束了局部图像块的数据分布统计。使用了一个用于超分辨率重构的新损失函数，结合了与训练人脸质量的重构误差作为对抗，使输出有更好的视觉效果。实验证明该方法在数据上和视觉上都达到了最先进水平。
 
 ![](https://cjmcv.github.io/deeplearning-paper-notes/images/pdSr/gln1.png)

Global Upsampling Network (GN)
 
GN网络有两条支路并行处理，在图像上采样的支路中使用反卷积层得到一个大的平滑而缺少细节的图像，使用双线性插值矩阵去初始化反卷积层权重，允许这些权重随着训练而更新，尽管反卷积层权重会更新，但不会更新太多使输出的是正常的平滑上采样图。全局细节生成的支路由全连接层作为编码层的网络实现，在除了最后用于生成128x128的上采样全局细节的层外，每一层的特征图都接ReLU。而且这里编码层无论是上采样4倍还是8倍都采用256维，这主要是因为训练样本有限，全局特征训练容易出现过拟合。最后将上采样的网络支路和全局细节特征生成支路的输出进行拼接，得到2x128x128的张量用于LN。


Local Refinement Network (LN)


LN的结构如图所示，分别对应上采样4倍和8倍任务，分析了三个有不同层数的全卷积网络框架。在每次卷积操作时都对图像做padding保持大小一致，卷积后输入到ReLU。全称没有使用池化，并且滑动步长为1，因此网络学习到了平移不变的非线性，如图2(c)，LN通过由GN得到的平滑和细节层，加强了人脸特定的局部特征。此外，重构图的局部数据分布与高分辨率对应图像块的数据分布相吻合（例如平滑的脸颊区域和尖锐的脸部轮廓）。

[info GAN](https://blog.csdn.net/wspba/article/details/54808833)
----


GAN，Generative Adversarial Network是目前非常火也是非常有潜力的一个发展方向，原始的GAN模型存在着无约束、不可控、噪声信号z很难解释等问题，近年来，在原始GAN模型的基础上衍生出了很多种模型，如：条件——CGAN、卷积——DCGAN等等，在本博客的前几篇博文里均进行了大致的解读，本篇博文将提到的InfoGAN也是GAN的一种改进成果，甚至被OPENAI称为去年的五大突破之一。

InfoGAN的出发点是，既然 GAN 的自由度是由于仅有一个 noise z，而无法控制 GAN 如何利用这个 z。那么我们就尽量去想办法在 “如何利用 z” 上做文章。于是 ,将 z 做了拆解，认为 GAN 中生成模型（G）应该包含的 “先验” 分成两种： （1）不能再做压缩的 noise z；（2）和可解释地、有隐含意义的一组隐变量 c_1, c_2, …, c_L，简写为 c 。这里面的思想主要是，当我们学习生成图像时，图像有许多可控的有含义的维度，比如笔划的粗细、图片的光照方向等等，这些便是 c ；而剩下的不知道怎么描述的便是 z 。这样一来，[7] 实际上是希望通过拆解先验的方式，让 GAN 能学出更加 disentangled 的数据表示（representation），从而既能控制 GAN 的学习过程，又能使得学出来的结果更加具备可解释性。为了引入这个 c ，[7] 利用了互信息的建模方式，即 c 应该和生成模型 （G）基于 z 和 c 生成的图片，即 G ( z,c )，高度相关 —— 互信息大。利用这种更加细致的隐变量建模控制，infoGAN 可以说将 GAN 的发展又推动了一步。首先，它们证明了 infoGAN 中的 c 对于 GAN 的训练是有确实的帮助的，即能使得生成模型（G）学出更符合真实数据的结果。其次，他们利用 c 的天然特性，控制 c 的维度，使得 infoGAN 能控制生成的图片在某一个特定语义维度的变化。

InfoGAN是什么 

简单的讲，就是一种常见的GAN，是在普通的GAN的基础上增加Q网络，可以通过无监督学习的方式学到生成的数据的类别。

二、小故事 

小D是一个很喜欢吃饺子的姑娘，喜欢吃不同的馅的饺子，而且对于饺子的要求十分高，尤其喜欢B城的一家饺子店的饺子，但是由于长期身在A城，没有办法吃到B城的饺子。 
而她的男朋友小G和她是异地，而且恰好是在B城，是一个非常宠她的小伙子，经常为了让她吃到满意的饺子，不断的尝试制作出和B城的饺子店一模一样的饺子。在每一次做完饺子之后，都会再去买一份小D爱吃的那家店的饺子，然后不辞千里给小D送去，让小D猜测哪个是他包的饺子，哪个是饺子店的饺子。终于，起初，小D总会一下子就能够分辨出来，终于，功夫不负有心人，有一天，小D已经分不出哪个是小G做的饺子哪个是饺子店的饺子了，因为它们都一样好吃了。 
故事讲到了这里，并没有结束，哈哈，毕竟好吃的除了男朋友，总少不了好闺蜜嘛，小D的好闺蜜叫小Q，不论是三观还是喜好都和小D保持高度一致，同样也喜欢吃B城的那家店的饺子，唯一不同的是，每次吃饺子的时候，都喜欢加点醋，但是，她拥有一个超能力，那就是，虽然不同馅的饺子的外形差异很细微，但是她是他们三个中唯一一个能够分辨出不同馅的饺子的样子的差异的人，而小D和小G并做不到这一点，经常吃一口才知道是什么馅的。 
好啦，故事讲到这里，就该结束了，其他细节请见下次分享。

三、InfoGAN网络结构 

1、判别器（Discriminator） 

小D：她的作用是判别饺子店的饺子和男朋友的饺子之间是否有差异的。 

而对于InfoGAN来说，就是判断real data （x）和生成器生成的fake data (G（z）)之间的差异有多少。 

2、生成器（Generator） 

小G：他的作用是不断的提高自己的造假能力，知道他做出的不同馅的饺子小D分辨不出来是他做的还是饺子店做的为止。

而对于InfoGAN来说，生成器就是利用噪声z和latent code c来进行生成仿真的数据，直到判别器无法分辨出数据到底是来自真实的数据x还是生成的数据G（z）为止。 

3、分类器 

小Q:她的作用是为了分辨出不同馅的饺子的差异，饺子上是没有标记的。 

而对于InfoGAN来说，Q网络是和D网络公用除了最后一层之外的其他所有的层的，它是为了分辨出数据之间的类别是什么，比如什么馅的饺子。 



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

[WGAN]( https://arxiv.org/abs/1701.07875)
----

[代码1](  https://github.com/hwalsuklee/tensorflow-generative-model-collections)

[代码2](  https://github.com/Zardinality/WGAN-tensorflow)

Wasserstein GAN（下面简称WGAN）成功地做到了以下爆炸性的几点：

彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度

基本解决了collapse mode的问题，确保了生成样本的多样性 

训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高（如题图所示）

以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到

![](https://pic1.zhimg.com/v2-3cfe84e6b6b58c00e013975fe649398e_1200x500.jpg)

与DCGAN不同，WGAN主要从损失函数的角度对GAN做了改进，损失函数改进之后的WGAN即使在全链接层上也能得到很好的表现结果，WGAN对GAN的改进主要有：

◆  判别器最后一层去掉sigmoid

◆  生成器和判别器的loss不取log

◆  对更新后的权重强制截断到一定范围内，比如[-0.01，0.01]，以满足论文中提到的lipschitz连续性条件。

◆  论文中也推荐使用SGD， RMSprop等优化器，不要基于使用动量的优化算法，比如adam，但是就我目前来说，训练GAN时，我还是adam用的多一些。

总的来说

GAN中交叉熵（JS散度）不适合衡量生成数据分布和真实数据分布的距离，如果通过优化JS散度训练GAN会导致找不到正确的优化目标，所以，WGAN提出使用wassertein距离作为优化方式训练GAN.虽然理论证明很漂亮，但是实际上训练起来，以及生成结果并没有期待的那么好。

[用于PyTorch中无监督异常检测的WGAN-GP](https://github.com/trigrass2/wgan-gp-anomaly)
----

[参考资料:不仅仅生成图片，用GAN做无监督的异常检测](https://zhuanlan.zhihu.com/p/32505627)

这篇文章要讲的不是用GAN来做图片的生成，而是一个非常有意思的应用方向-把GAN应用于做异常检测。

![](https://pic4.zhimg.com/80/v2-3afb9ef54ee08b8983abcec88975ff17_hd.jpg)

![](https://pic1.zhimg.com/80/v2-592e8ddc687eda45212d8ddf70853aeb_hd.jpg)

这个框架图基本已经解释了这篇paper的方法，其实就是用正常的图片去训练GAN，然后通过GAN生成与异常图对应的正常图来对比找到异常。

paper中使用的GAN的结构图，其实在这篇paper里面使用的GAN就是普通的DCGAN，从一个噪声向量Z学习生成一张图片。我们可以看到正常的眼部OCT图的纹理是一种比较正常的过渡，但是异常的OCT图明显纹理产生了变化。DCGAN只用正常的OCT图像训练，这样DCGAN就只能从噪声生成正常纹理的OCT图像。当输入一个异常图时，通过比较DCGAN的生成图和异常图的差异去更新输入的噪声Z，从而生成一个与异常图尽可能相似的正常图。通过这样的方式，可以认为重建出了异常区域的理想的正常情况，这样两张图一对比不仅仅可以认定异常情况，同时还可以找到异常区域。
![](https://pic4.zhimg.com/80/v2-9dbf3f09ea6ba47de03569ad1ad0bd6e_hd.jpg)

[ST-CGAN 用GAN实现阴影检测和阴影去除](https://arxiv.org/abs/1712.02478)
----

![ST-CGAN的模型结构](https://pic4.zhimg.com/80/v2-686a6c78c74aeaac1e65fdb2e772a912_hd.jpg)

由上图我们可以看到，ST-CGAN构建了两个生成器，G1用于从原图生成含有阴影的图，G2用于从原图和阴影图的叠加中生成去除了阴影的图，而两个判别器则分别监督这两个生成过程，最终达到收敛。

![](https://pic4.zhimg.com/80/v2-fd4b5eb495c4684f3ca909adcc5dd83e_hd.jpg)

优点：

1.idea很不错，用GAN来做Detection和Removal，为大家打开了思路，现在有不少人都在思考怎么把GAN拓展到更加广的应用场景里。

2.实验效果不错，与几个Baseline相比效果有明显提升。

缺点：

1.正如文中所说，论文描述里一些地方指代不清，希望原作者后续能给出新的版本。

2.相对于其它模型，ST-CGAN所需要的先验条件变多了，具体来讲就是需要包含阴影的原始数据，阴影图，不包含阴影的原始数据这三类数据，而其它的模型是在不具备第三类数据的情况下做的，在这样的情况下ST-CGAN表现更好可能跟它获取的输入条件更多有关。

3.实验中给出了阴影部分教深的情况下的实验结果，并没有给出在阴影部分较浅的情况下的结果，其次如何给出一个深浅的标准也是一个值得讨论的问题，这可能直接影响到对模型评价标准。


[用于图像转换的感知对抗网络 PAN]()
----

摘要

在本文中，我们提出了一种用于图像转换任务的原理感知对抗网络（Perceptual Adversarial Network，PAN）。与现有算法不同——现有算法都是针对具体应用的，PAN 提供了一个学习成对图像间映射关系的通用框架（图1），例如将下雨的图像映射到相应的去除雨水后的图像，将勾勒物体边缘的白描映射到相应物体的照片，以及将语义标签映射到场景图像。

本文提出的 PAN 由两个前馈卷积神经网络（CNN）、一个图像转换网络T 和一个判别网络D组成。通过结合生成对抗损失和我们提出的感知对抗损失，我们训练这两个网络交替处理图像转换任务。其中，我们升级了判别网络 D 的隐藏层和输出结果，使其能够持续地自动发现转换后图像与相应的真实图像之间的差异。

同时，我们训练图像转换网络 T，将判别网络 D 发现的差异最小化。经过对抗训练，图像转换网络T 将不断缩小转换后图像与真实图像之间的差距。我们评估了几项到图像转换任务（比如去除图像中的雨水痕迹、图像修复等）实验。结果表明，我们提出的 PAN 的性能比许多当前最先进的相关方法都要好。
![](https://img-blog.csdn.net/2018032716512413?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Vkb2dhd2FjaGlh/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

项目代码库
----

[生成敌对文本到图像合成](http://www.paperweekly.site/papers/996)
----

摘要

从文本描述合成高质量的图像是计算机视觉中的一个挑战性问题，并且具有许多实际应用。通过现有的文本到图像方法生成的样本可以粗略地反映给定描述的含义，但它们不包含必要的细节和生动的对象部分。在本文中，我们提出Stacked Generative Adversarial Networks（StackGAN）来生成256x256照片写实图像，其条件是文本描述。我们通过草图细化过程将难题分解为更易处理的子问题。Stage-I GAN根据给定的文字描述描绘对象的原始形状和颜色，产生Stage-I低分辨率图像。Stage-II GAN将第一阶段的结果和文字描述作为输入，并生成具有照片逼真细节的高分辨率图像。它能够纠正第一阶段结果中的缺陷，并通过细化过程增加引人注目的细节。为了改善合成图像的多样性并稳定条件GAN的训练，我们引入了一种新的调节增强技术，该技术鼓励潜在调节流形中的平滑性。对基准数据集进行大量的实验和与现有技术的比较表明，所提出的方法在生成以文本描述为条件的照片拟真图像方面取得显着改进。我们引入了一种新颖的调节增强技术，可以在潜在的调节歧管中提高光滑度。对基准数据集进行大量的实验和与现有技术的比较表明，所提出的方法在生成以文本描述为条件的照片拟真图像方面取得显着改进。我们引入了一种新颖的调节增强技术，可以在潜在的调节歧管中提高光滑度。对基准数据集进行大量的实验和与现有技术的比较表明，所提出的方法在生成以文本描述为条件的照片拟真图像方面取得显着改进。

[VAE]()
----

[VAE实现代码（caffe）](https://github.com/cdoersch/vae_tutorial)

[vae (keras)](https://github.com/bojone/vae)

生成模型和判别模型组合的另一个套路——Variational autoencoder，简称VAE。

![](https://pic3.zhimg.com/80/v2-42e6db06401f563ad5cce49eb2a56d67_hd.jpg)

VAE的想法是通过对编码器的输出做出改变，如下图所示。

![](https://pic3.zhimg.com/80/v2-e92ff7bb419fff3b906dfce5fc550820_hd.jpg)

![](https://pic2.zhimg.com/80/v2-38ff43119cf9f11f4e42c6e7c18982e7_hd.jpg)

VAE为什么work？一种直觉的想法是，其是加上噪声了的。比如上图中，要在一张满月图和半月图中生成一张图，加上了噪声后，能够使得在很大范围内都是月亮的圆和半圆的形状，所以能够在一定程度上保证在其中间的生成图是介于两者之间的。

其主要数学原理是高斯混合模型（Gaussian Mixture Model）。

![](https://pic4.zhimg.com/80/v2-433a704d0c7197b971b9f7e94202d040_hd.jpg)

VAE的问题是其实VAE并没有想过怎样去生成一张新图，而是在产生一张与数据库中的图尽可能相似的图，只是在模仿。上图中一个像素点的不同在原始的“7”图上，左边的可以理解，但是右图是不一样的，是假的，但是VAE会认为说其与原始目标是一致的。

VAE并没有生成新图，而是仅仅记住了已经存在了的图。


基础概念部分：
 
KL divergence，翻译过来叫做KL散度。

KL散度，他可以很好地测量两个概率分布之间的距离。如果两个分布越接近，那么KL散度越小，如果越远，KL散度就会越大。

KL散度的公式为：

![](https://www.zhihu.com/equation?tex=KL%28p%7C%7Cq%29%3D%5Csum%7Bp%28x%29log%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D%7D)这个是离散概率分布的公式，
![](https://www.zhihu.com/equation?tex=KL%28p%7C%7Cq%29%3D%5Cint%7Bp%28x%29log%7B%5Cfrac%7Bp%28x%29%7D%7Bq%28x%29%7D%7Ddx%7D)这个是连续概率分布的公式
这个是离散概率分布的公式，

KL(p||q)=\int{p(x)log{\frac{p(x)}{q(x)}}dx}，这个是连续概率分布的公式


[Ragan](https://zhuanlan.zhihu.com/p/39050343)
----

[RaGan代码](https://github.com/AlexiaJM/RelativisticGAN)

在论文中，她指出现有的标准GAN（SGAN）还缺少一个基本属性，即训练生成器时，我们不仅应该提高伪数据是真实数据的概率，还应该降低实际数据是真实数据的概率。这个属性是一个重要基础，它也是所有GAN都应该遵守的。

摘要
在标准生成对抗网络（SGAN）中，判别器负责估计输入数据是真实数据的概率，根据这个数值，我们再训练生成器以提高伪数据是真实数据的概率。但本文认为，判别器在提高“伪数据为真”的概率的同时，也应该降低“实际数据为真”的概率，原因有三：

mini-batch中一半的数据是伪数据，这个先验会带来不合逻辑的结果；

在最小化散度（divergence minimization）的过程中，两个概率不是同步变化；

实验证实，经过相对判别器诱导，SGAN的性能可以媲美基于IPM的GAN（WGAN、WGAN-GP等），而后者实际上已经具有相对判别器的雏形，因此也更稳定。

本文提出相对GAN（RGAN），并在它的基础上又提出了一个变体——相对均值GAN（RaGAN），变体用平均估计计算判别器概率。此外，论文还显示基于IPM的GAN其实是RGAN的子集。

通过比较，文章发现：(1)相比非相对GAN，RGAN和RaGAN更稳定，产出的数据样本质量更高；(2)在RaGAN上加入梯度惩罚后，它能生成比WGAN-GP质量更高的数据，同时训练时长仅为原先的1/5；(3)RaGAN能够基于非常小的样本（N = 2011）生成合理的高分辨率图像（256x256），撇开做不到的GAN和LSGAN，这些图像在质量上也明显优于WGAN-GP和SGAN生成的归一化图像。

实验对比
简而言之，相对的GAN和普通GAN的区别如下所示。

标准GAN（SGAN）的判别器：

![](https://pic1.zhimg.com/80/v2-1984f10a1942c1647ae890e93d8fe2b2_hd.jpg)

相对标准GAN（RSGAN）的判别器：

![](https://pic3.zhimg.com/80/v2-95c2707a1c24f4cdec2063daa54d9709_hd.jpg)

相对均值标准GAN（RaSGAN）的判别器：

![](https://pic1.zhimg.com/80/v2-aab4b1b5ea5b38fea18a36a7aca13828_hd.jpg)

![](https://pic1.zhimg.com/80/v2-1af077c56baef833c9b2ad0c1e94ef3b_hd.jpg)

可以发现，在无需额外计算成本的前提下，相对的判别器显著提高了数据的质量和稳定性，而且比现有GAN效果更好


BEGAN（边界均衡生成对抗网络）(https://zhuanlan.zhihu.com/p/26394806)
----

### 1. BEGAN简介

![](https://pic2.zhimg.com/80/v2-ad7bed2e916c7d55e6180efcfd3271f4_hd.jpg)

作者通过实验发现，每个pixel的重构误差实际上是独立同分布的，并且都是（近似）正态分布。根据中心极限定理，整个图像的重构误差也将服从相同的正态分布。

据此，作者提出了让生成图像的重构误差分布逼近真实图像的重构误差分布的做法，而传统的GAN的做法是让生成图像的分布逼近真实图像的分布。


[code](https://github.com/Heumi/BEGAN-tensorflow)


TPGAN(侧脸生成正脸)(https://github.com/Heumi/BEGAN-tensorflow)
----

[code](https://github.com/HRLTY/TP-GAN)



###### Fully convolutional adaptation networks for semantic segmentation 无监督语义分割之全卷积域适应网络

为了获得像素级语义分割数据集，该文作者设计一网络，从游戏引擎获取语义分割图像，并处理使其能够用于真实世界的语义分割训练。


解决思路

针对合成图像和真实图像之间的域适应问题，本文主要提出了两种域适应策略，

分别是图像层面的域适应（Appearance Adaptation）和特征表示层面的域适应（Representation Adaptation），

具体实现为两个网络架构：图像域适应网络（Appearance Adaptation Networks，AAN)和特征适应网络（Representation Adaptation Networks，RAN)。

整体网络架构如下图所示：


![](http://static.extremevision.com.cn/donkey_7ccb8394-0fce-43d3-b4b1-1a22d441be9b.jpg)

![](http://static.extremevision.com.cn/donkey_26b72371-ef26-4286-90c3-970b42720b61.jpg)

![](http://static.extremevision.com.cn/donkey_9c36f782-2cf1-4599-a6c0-a7a928f7cc1f.jpg)

总结展望

本文贡献：

（1）提出了语义分割任务中的域适应问题：如何利用合成数据有效提升真实场景中的语义分割性能；

（2）提出了两个层面的域适应策略（图像层面的域适应和特征层面的域适应），用于解决该问题。

###### ETH Zurich提出利用对抗策略，解决目标检测的域适配问题

简而言之：使用对抗神经网络，将另一数据集的图像转换为和目标数据集近似的图像

![](http://static.extremevision.com.cn/donkey_df1e099a-f9cd-4dc7-895a-823e2a527863.jpg)

域适配（Domain Adaptation, DA）问题已经在图像分类任务上得到了广泛研究并取得了惊人进展，今年CVPR上也有很多相关工作。其本质属于迁移学习的一种，问题设定是：如何使得源域（Source Domain）上训练好的分类器能够很好地迁移到没有标签数据的目标域上（Target Domain）上。

其中两个代表性的工作有：DSN [1]和ADDA [2]。

解决思路

本文首先从概率分布的角度论证了进行域适配的必要性，据此引出了本文的两点贡献：

提出图像层面的适配（Image-Level Adaptation）和目标层面的适配（Instance-Level Adaptation），

用于解决自动驾驶场景下目标检测任务使用不同数据训练的域适配问题。

![](http://static.extremevision.com.cn/donkey_cfdfc97f-8e2c-4466-a772-6e16389b62b6.jpg)

图2给出了本文方法的处理流程，检测器采用的是当前主流的Faster R-CNN，训练时一个batch包含两张图像，分别来自源域和目标域，所以网络的输入实际上是两张图像（图中只画了一张）。

结合图2， 

下面重点论述下本文是所采用的两个层面的域适配：

1、Image-Level Adaptation

源域的图像是有标签的，而目标域的图像是无标签的，如果只用源域的图像进行训练，会使得网络学到的特征在源域上非常discriminative，但在目标域上表现欠佳，所以应该做图像层面上的适配，也即使得来自源域的图像特征和来自目标域的图像特征满足同一分布，这里可以采用图2中的Image-level domain classifier，如果使得domain classifier无法区分开特征图到底是来自源域还是来自目标域，那么我们的目的就达到了，而在训练过程中domain classifier则是要尽量把二者区分开，其实这里体现的就是GAN中的对抗策略了，只不过作者用的并不是当前火热的GAN，而是通过文献[3]提出的gradient reverse layer实现的


2、Instance-Level Adaptation

同样，我们还需要目标层面的适配，也就是每个ROI的特征也要满足同一分布，同上面的对抗策略一致


