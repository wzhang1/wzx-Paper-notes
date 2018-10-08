lanenet代码阅读:
========================================================

tensorflow 版
---------------------------------------

[代码](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

系统配置：  GTX 1080 Ti

tensorflow-gpu v1.3.0 | cuda8.0 |  cuDNN 6.0 |

[tensorflow版本与cuda cuDNN版本对应使用](https://blog.csdn.net/lifuxian1994/article/details/81103530)


项目进展记录：
------------------------

初步确认输入没问题 通过test input 代码在 train_lanenet_wzxtest.py



问题：
-----------------------------

1.gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]  ：为什么要减掉这个值

试验后：去掉这句话后，训练和测试准确率大降



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




