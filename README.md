# 三层神经网络分类

## 实验介绍

使用MLP对Fashion-MNIST进行拟合，实验要求不使用自动微分框架，使用numpy支持的矩阵乘法，手动实现反向传播，实现mlp参数更新，实验涉及如下

* 隐藏层层数
* 学习率退火
* 最优超参寻找

## 数据集介绍

Fashion-MNIST数据集，图片类型数据，图片存储为(1, 28, 28)，数据如下

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="/Users/lizhiwei/Documents/code/cls/fmnist.png" alt="Image" style="width: 90%; margin: 10px;">
</div>

## 反向传播公式

以两层隐藏层为例，简单记一层参数分别为 $W_1$ 和 $b_1$，第二层参数分别为 $W_2$和 $b_2$，Softmax为 $softmax$，Relu为$ Relu$，那么两层隐藏层的MLP的预测结果可以表示为

$$\hat{y} = softmax(W_2(Relu(W_1X + b_1)) + b_2)$$

该任务采用损失函数为交叉熵损失函数，形式为

$$l = -\sum y_i\ log\ \hat{y}_i$$


使用中间变量 $z = W_2(Relu(W_1X + b_1)) + b_2$，于是 $\hat{y} = softmax(z)$使用链式法则求导，那么第二层梯度为

$$\frac{\partial l}{\partial W_2} = \frac{\partial l}{\partial \hat{\textbf{y}}}\cdot\frac{\partial \hat{\textbf{y}}}{\partial z}\cdot\frac{\partial z}{\partial W_2}$$

$$\frac{\partial l}{\partial b_2} = \frac{\partial l}{\partial \hat{\textbf{y}}}\cdot\frac{\partial \hat{\textbf{y}}}{\partial z}\cdot\frac{\partial z}{\partial b_2}$$

使用中间变量 $s = W_1X+b_1$和 $t=Relu(s)$，于是 $z=W_2t+b_2$，那么第一层梯度为

$$\frac{\partial l}{\partial W_1} = \frac{\partial l}{\partial \hat{\textbf{y}}}\cdot\frac{\partial \hat{\textbf{y}}}{\partial z}\cdot\frac{\partial z}{\partial t}\cdot\frac{\partial t}{\partial s}\cdot\frac{\partial s}{\partial W_1}$$

$$\frac{\partial l}{\partial W_1} = \frac{\partial l}{\partial \hat{\textbf{y}}}\cdot\frac{\partial \hat{\textbf{y}}}{\partial z}\cdot\frac{\partial z}{\partial t}\cdot\frac{\partial t}{\partial s}\cdot\frac{\partial s}{\partial b_1}$$

那么其中需要计算的部分如下

$$\frac{\partial l}{\partial \hat{\textbf{y}}}\cdot\frac{\partial \hat{\textbf{y}}}{\partial z}=\hat{\textbf{y}} - \textbf{y}$$

这里利用softmax和cross entropy的很好的性质推的，具体需要写出具体矩阵元素

那么

$$\frac{\partial l}{\partial W_2} = (\hat{\textbf{y}} - \textbf{y})\cdot \frac{\partial z}{\partial W_2} =  (\hat{\textbf{y}} - \textbf{y})\cdot Relu(W_1X+b_1)^T$$

$$\frac{\partial l}{\partial b_2} = (\hat{\textbf{y}} - \textbf{y})\cdot \frac{\partial z}{\partial b_2} =  (\hat{\textbf{y}} - \textbf{y})$$

$$\frac{\partial l}{\partial W_1} = (\hat{\textbf{y}} - \textbf{y})\cdot W_2^T\cdot Relu^{'}(W_1X+b_1)\cdot X^T$$

$$\frac{\partial l}{\partial b_1} = (\hat{\textbf{y}} - \textbf{y})\cdot W_2^T\cdot Relu^{'}(W_1X+b_1)$$

其中 $Relu^{'}(\cdot)$为对 $Relu$的导数，可以视为

$$
f(x) = 
\begin{cases} 
1 & \text{if } x > 0 \\
0 & \text{if } x < 0 
\end{cases}
$$


## 训练结果

实验设置采用余弦退火的学习率，隐藏层数对实验影响如下

<div style="display: flex; justify-content: space-between;">
  <img src="/Users/lizhiwei/Documents/code/cls/lr.png" alt="Image 1" style="width: 46%; margin: 10px;">
  <img src="/Users/lizhiwei/Documents/code/cls/test_acc_combine.png" alt="Image 2" style="width: 46%; margin: 10px;">
</div>

<div style="display: flex; justify-content: space-between;">
  <img src="/Users/lizhiwei/Documents/code/cls/train_loss.png" alt="Image 3" style="width: 46%; margin: 10px;">
  <img src="/Users/lizhiwei/Documents/code/cls/val_loss.png" alt="Image 4" style="width: 46%; margin: 10px;">
</div>

## 模型调优

采用三层隐藏层进行参数调优，针对隐藏维度、学习率、正则化强度进行最优参数搜索，结果如下

| hidden_size | learning_rate | regularization_strength  | accuracy |
|-------------|---------------|--------------------------|----------|
| 64          | 0.4           | 0.0                      | 81.2     |
| 64          | 0.4           | 0.01                     | 80.9     |
| 64          | 0.35          | 0.0                      | 82.1     |
| 64          | 0.35          | 0.01                     | 82.3     |
| 64          | 0.3           | 0.0                      | 80.1     |
| 64          | 0.3           | 0.01                     | 83.1     |
| 128         | 0.4           | 0.0                      | 82.3     |
| 128         | 0.4           | 0.01                     | 82.5     |
| 128         | 0.35          | 0.0                      | 80.8     |
| 128         | 0.35          | 0.01                     | 80.9     |
| 128         | 0.3           | 0.0                      | 82.5     |
| 128         | 0.3           | 0.01                     | 84.3     |
| 256         | 0.4           | 0.0                      | 79.2     |
| 256         | 0.4           | 0.01                     | 80.1     |
| 256         | 0.35          | 0.0                      | 81.9     |
| 256         | 0.35          | 0.01                     | 80.7     |
| 256         | 0.3           | 0.0                      | 80.9     |
| 256         | 0.3           | 0.01                     | 81.3     |

经过超参数搜索后发现，隐藏维度为128，学习率为0.3，正则化强度为0.01的效果最好，达到84.3

