# 神经网络实现鸢尾分类

### 准备数据

- 数据集读入
  - sklearn库中的iris数据集
  - 分别读入feature 和 label

- 数据集乱序
  - 通过random随机排序
  - 120 / 30 划分数据

- 生成训练集和测试集（x_train/y_train ， x_test/ y_test)
- 配成（输入特征，标签）对，每次读入一个batch
  - 由于随机中使用相同的seed特征与标签相互对应


### 搭建网络

- 定义神经网络中所有可训练参数
  - iris数据集共4个特征点，所以采用4个节点的输入层
  - iris数据集共3个结果，所以采用3个节点的输出层
  - 本网络未包含隐含层


### 参数优化

- 嵌套循环迭代，with结构更新参数，显示当前loss

  ```python
  with tf.GradientTape() as tape:
      若干个计算过程
  grad = tape.gradient(函数，对谁求导)
  ```

  

### 测试效果

- 计算当前参数前向传播后的准确率，显示当前acc
- 使用matplotlib 中的pyplot绘制loss与accuracy的图像

