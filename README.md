# 基于深度学习的定长验证码识别
## 项目介绍
本项目的数据集完全使用python生成，生成长度为5位，字符包涵所有大写字母和0-9数字。数据集大小完全可以通过配置文件设定，
从而能够让不同配置的设备都能从容选择。项目的目标是使得准确率能够达到95%以上。
但是丑话写在前头，目前最好的识别效果为80%左右，实验结果如下，且该识别率在相同的条件下无法复现，所以运气占了绝大多数。
等以后有时间了再精进
![img.png](img.png)
## 文件介绍
### generate_verification_code.py
该文件实现验证码的生成，验证码图片的大小固定为160 * 80，这是为了使得像素的大小在模型中经过变化之后仍为整数。
分别有四种模式，1代表生成训练数据集，2代表生成测试数据集，3代表删除训练数据集，4代表删除测试训练集

生成的数据集格式是这样的：图片分别存储在对应的文件夹下，图片的名称分别为序号，而文件夹外的txt文件中，
则存储着序号对应图片对应的标签。

### CustomDataset.py
该文件负责实现数据集读取的功能，因为数据集由我们自己设计，因此读取方法也得由我们来实现，
该文件下面的类继承了torch的Dataset类，然后实现了读取图片和标签的功能。并且进行了预处理，变为了tensor，并且灰化使得图片的维度只有一层

在该项目下，读取完的label需要转换为tensor，五个字符，每个字符有36种可能，因此tensor为5 * 36的二位向量，
在36个位置中，在字符对应的位置取1，如字符顺序为“ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789”，那么字符为D时，
索引为3的位置取1。

除此之外，还实现了两个方法用来将tensor转为str，或是将str转为tensor

### models.py
该文件负责实现神经网络模型的搭建。网络的前四层是先使用CNN进行提取，然后使用激活函数，并通过池化层减小宽度和长度。
到达最后一层时，先使用view函数将tensor展平，最后使用全连接层进行概率输出。

最终判断的时候是将tensor分为五段，变成二维的tensor，在每一段中选取概率最高的作为字符输出。

### config.py
该文件负责该项目的配置项的编辑、如迭代次数、训练集大小等等内容。

### __main__.py
该文件负责主要的训练和测试过程。