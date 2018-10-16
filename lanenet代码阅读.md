lanenet代码阅读:
========================================================

tensorflow 版
---------------------------------------

[代码](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

系统配置：  GTX 1080 Ti

tensorflow-gpu v1.3.0 | cuda8.0 |  cuDNN 6.0 |

[tensorflow版本与cuda cuDNN版本对应使用](https://blog.csdn.net/lifuxian1994/article/details/81103530)

运行命令：
----------

python train_lanenet.py --net vgg --dataset_dir data/training_data_example/

python train_lanenet.py --net enet --dataset_dir data/training_data_example/

python test_lanenet.py --is_batch False --batch_size 1 --weights_path model/culane_lanenet/culane_lanenet_vgg_2018-10-09-09-10-01.ckpt-2000 --image_path data/tusimple_test_image/0.jpg

python train_lanenet_wzxtest.py --net enet --dataset_dir data/training_data_example/ --weights_path model/culane_lanenet/culane_lanenet_1_enet_2018-10-09-10-56-16.ckpt-7200

python readtest.py --net enet --dataset_dir testdata --weights_path model/culane_lanenet/culane_lanenet_enet_2018-10-09-18-07-49.ckpt-64800

tensorboard --logdir=/media/wzx/AE5A299C5A2961F7/lanenetwzx/lanenet-lane-detection-master/tboard/culane_lanenet/enet

python readtest2.py --net enet --dataset_dir testdata --weights_path model/culane_lanenet/culane_lanenet_enet_2018-10-09-18-07-49.ckpt-64800





项目进展记录：
------------------------

初步确认输入没问题 通过test input 代码在 train_lanenet_wzxtest.py

单支训练左边分支效果很好，下一步，增加训练集增强其泛化能力、研究其能预测到被遮挡车道线的原因、对结构进行优化，加速或加残差。

初步运行调试成功进入测试环节

测试结果:

![](https://github.com/greenfishflying/wzx-Paper-notes/blob/master/image/43.png)

![](https://github.com/greenfishflying/wzx-Paper-notes/blob/master/image/530.png)


正在增加训练集对模型的泛化能力进行提升

数据扩增：添加高斯噪声，色彩饱和度等扩增，弯道，左右翻转（还没做）






问题：
-----------------------------

1.gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]  ：为什么要减掉这个值

试验后：去掉这句话后，训练和测试准确率大降

2.test 模块，enet的test不能用，其主要语句在
 net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
 但是换成enet之后不可用
 
3.为什么enet能预测到被遮挡的车道线?

为了构建GT分割图，我们将每天车道线的对应像素连成线，这么做的好处是即使车道线被遮挡了，网络仍能预测车道位置。

4.lanenet是先分割再聚类？


5.densenet是什么

DenseNet提出了一个更激进的密集连接机制：即互相连接所有的层，具体来说就是每个层都会接受其前面所有层作为其额外的输入。

![](https://upload-images.jianshu.io/upload_images/11692737-7b1b6ef5b02ec02b?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)



主干：本机上代码由于所有代码都放进了tools里面，故，结构与原网页不同
-----------------------------------------------------------

train_lanenet.py

def init_args: 输入并储存 数据集地址、网络类型、预训练权重

def train_net:

设置变量{

        训练数据

        验证数据

        初始化参数：

        输入?

        二值化标签

        分割标签

        超参数

        计算loss

        计算accuracy

        存储权重
}

设置sess{

        各种设置，还没看懂

}


with sess.as_default():创建神经网络会话
        
各种参数的设置：
        {
        
        如果没输入预训练权重：
        
            init = tf.global_variables_initializer() ：初始化参数
            
            sess.run(init) ：运行
            
        否则：
        
            准备读参数
            
        加载预训练参数
        
        如果是vgg
        
        加载vgg权重
        
  }
        
        
 for 轮训练
        {
        
           读图并resize：
           
           gt_imgs :原图
           
           binary_gt_labels：二值分割图（左）
           
           instance_gt_labels： 语义分割图（右）
           
           
           开始训练 print("training")
           
           _, c, train_accuracy, train_summary, binary_loss, instance_loss, embedding, binary_seg_img = sess.run（）
           
           每10轮：
           
           {
           输出图像：
           
           image ：归一化后的原图
           
           binary_label：二值标签图（ground truth）
           
           instance_label 实例分割图（ground truth）
           
           binary_seg_img ：Binary Segmentation Image 二值分割图像（左边输出）
           
           embedding ：Instance Embedding Image 实例嵌入图像（右边输出）
           
           }
           
           验证部分：
           
           {
           
           读图（验证集）
           
           验证（不训练）
           sess.run([total_loss, val_merge_summary_op, accuracy, binary_seg_loss, disc_loss],
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label: binary_gt_labels_val,
                                    instance_label: instance_gt_labels_val,
                                    early_drop_prob: 0,
                                    later_drop_prob: 0,
                                    phase: phase_val})
           
        }
        
        每100轮输出一次测试图片
        
        每200轮保存一次权重
        
}




