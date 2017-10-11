# key-value-MemNN

这是论文“Key-Value Memory Networks for Directly Reading Documents”的tensorflow实现方案，使用的数据集是MovieQA，基于KB知识库作为知识源进行建模。

其结构图如下所示：

![](https://i.imgur.com/9kpH8sY.png)

首先我们看一下模型仿真出来的网络架构，如下图所示：

![](https://i.imgur.com/vkhSM7o.png)

但是模型的仿真效果比较差，如下图所示，acc一直上不去，而且loss维持在10附近居高不下。到了训练后期

到了训练后期loss反而逐渐升高，模型无法收敛，如下图所示==

![](https://i.imgur.com/CtukUD9.png)

观察了模型运行过程中的一些参数，发现原因可能出现在B矩阵的身上，如下图，在训练后期也初夏大规模的震荡。所以对模型做出了改进：

![](https://i.imgur.com/3MOZrMe.png)

改进的方案是添加一个bias，来减少B矩阵的偏置。此外，去看了FaceBook官网给出的模型训练的参数，发现embedding-size取得500，

而且，max_slots取得1000还是多少，所以尝试按照其参数进行训练（发现训练速度极慢无比）。

所以最后作出的调整是：

1，给B矩阵添加bais
2，进行梯度截断，截断值去10
3，将max_slots增大至300
4，将embedding_size增大至200

按照上述方案进行训练，速度还可以接受，而且在模型收脸上也收到了很好的效果，如下图所示：

![](https://i.imgur.com/1V9naa8.png)

而且B矩阵虽然仍有尖峰存在，但是相对而言已经减少了很多，如下

![](https://i.imgur.com/UAs0U4Q.png)

训练过程的输出如下所示，结合上图可以发现，最起码训练集上的acc可以到达0.8~0.9，loss也维持在1附近。
但是仍然存在缺点就是测试集上的效果还不够好，准确度只有0.3左右。这是接下来要进行解决的问题：
![](https://i.imgur.com/3a43lWe.png)
