# 机器学习基础知识
【机器学习】：任务T  性能度量方案P/ 计算机程序：自主学习任务T的经验E

## 机器学习算法
| 类型 | 传统算法(Traditional Algorithm) | 学习算法(Learning Algorithm) |
|-------|-------------------------------|------------------------------|
| **原理** | 人工定义规则，按步骤执行 | 从数据中学习模式和规律 |
| **数据依赖** | 固定逻辑，不依赖数据 | 需要大量数据来学习 |
| **示例** |排序算法(Bubble Sort)、搜索算法(Binary Search)| 线性回归(Linear Regression)、神经网络(Neurak Netwoork) |
| **使用场景** | 规则明确的任务，如计算税费、排序 | 复杂问题，如语音识别、图像分类 |
## 机器学习算法类型
### 监督学习 (Supervised Learning, SL)
使用标记输入输出已知的样本数据。主要解决两类问题：回归(regression)和分类(classification)
x --> y(y given)
- regression
  - predict a number from infinitely many possible outputs
    - {housing price prediction, x = house size(ft2), y = price($),
    - 从散点图里画regression curves}
- classification
  - only n possible output (n = # of categories)
    - {肿瘤诊断：x = tumor size(cm), y = diagnosis(0/1), 分类，良性O，恶性X（可以有multiple output，如恶性1类 X， 恶性2类 V, 即multiple categories,如（猫/狗））
    - more input to predict output
      - i = tumor size, j = age, 分类：良性O，恶性X
      - the learning algorithm has to decide how to fit a boundary line.
#### 常用算法
    - 邻近算法 (K-Nearest Neighbors, KNN)
    - 线性回归 (Linear Regression)
    - 逻辑回归 (Logistic Regression)
    - 支持向量机 (Support Vector Machine, SVM)
    - 朴素贝叶斯分类器 (Naive Bayes)
    - 决策树 (Decision Tree)
    - 随机森林 (Random Forests)
    - 神经网络 (Neural Network)
        - 卷积神经网络 (Convolutional Neural Networks, CNN)
        - 深信度网络 (Deep Belief Network, DBN)

### 无监督学习 (Unsupervised Learning, UL)
使用未标记的样本数据，在未知输入和预先确定的输出之间建立连接。主要解决：输入数据聚类(Clustering)和输入特征变量关联(Correlation)。
#### 常用算法
    - K均值聚类 (K-Means Clustering)
    - DBSCAN (Density-based Spatial Clustering of Applications with Noise)
    - 主成分分析算法 (Principal Component Analysis, PCA)
    - 自组织映射神经网络 (Self-Organizing Map, SOM)
    - 受限玻尔兹曼机 (Restricted Boltzmann Machine, RBM)

### 强化学习 (Reinforcement Learning, RL)
在变化的环境中通过试错学习，根据规则作出最佳决策。应用实例：下棋机器人、无人驾驶。


## Python相关基础
### 重要包
1. **Numpy**: 数值运算库
         - 许多库的基础

2. **Pandas**: 数据处理库
         - 支持多种文件格式导入(CSV, JSON, SQL, Excel)
         - 主要数据结构：Series, DataFrame

3. **SciPy**: 科学计算库
         - 基于Numpy构建
         - 包含数据处理、优化、插值、线性代数等模块

4. **Matplotlib**: 基础可视化库

5. **Scikit-learn**: 机器学习库
         - 免费开源
         - 支持监督和非监督算法
         - 与NumPy, SciPy, Matplotlib, Pandas集成


## 核心组件

### 多层感知机（MLP）
#### 神经网络结构
神经网络结构指人工神经网络（**ANN**, Artificial Neural Network)的**层级组织方式**，即**神经元（节点）如何排列和链接**形成一个可以进行计算的系统。
你可以把**神经网络**想像成一个模拟人脑神经元工作方式的计算模型，它由多个**层(layers)** 组成，每一层包含若干**神经元(Neurons)**，这些神经元通过**权重(Weights)** 和**激活函数(ACtivation functions)** 进行信息传递和处理。
- **输入层（Input Layer）**: 负责接收数据，如一张图片的像素值或一组数值的特征。
- **隐藏层（Hidden Layers)**： 进行计算，负责学习数据的特征。
                           可能有多个隐藏层，层数越多，网络越深。
                           每个隐藏层的神经元会通过权重和激活函数处理输入数据。
- **输出层（Output Layer)**：负责输出最终的预测结果，如分类任务的类别或回归任务的数值。
多层感知机（MLP,Multi-Layer Perceptron）是最基础的神经网络结构。每一层都是一组神经元，每个神经元与前一层的所有神经元相连。
#### 权重(Weights)
- **权重是神经网络中的核心参数，用于决定每个神经元输入的重要性。** 通过不断调整来优化模型的性能。

- 在一个神经网络中，每个神经元会接收**来自上一层神经元的输入**，然后进行**加权求和**。
  假设某个神经元有n个输入x1,x2,...,xn,它的计算公式是：
  z = w1x1 + w2x2 + ... + wnxn + b
  xi:输入的数据
  wi:对应的权重
  b:偏置(bias),用于调整输出
  z:加权和的结果，随后作为**激活函数**的输入

- **权重的优化**：神经网络的学习过程就是不断调整权重w以最小化损失。
  随机初始化权重（通常使用小的随机值）
  计算**向前传播**
  计算**损失**
  通过**反向传播**调整权重
  重复训练，直到找到合适的权重
#### 激活函数(Activation Function)
- 激活函数是神经网络中的**非线性变换函数**，用于决定神经元是否“激活”（输出非零值）。
  **它的作用是引入非线性，使神经网络能学习复杂的模式**
  如果神经网络没有激活函数，每一层只对输入做线性变换（输出=WX+b），无论多少层，最终仍然是线性函数。

- **常见的激活函数**
- ReLU(Rectified Linear Unit,修正线性单元)
  - ![ReLU](https://latex.codecogs.com/svg.image?&space;f(x)=\max(0,x))
  - 优点：简单计算，不会出现梯度消失问题
  - 缺点：可能会出现“神经元死亡”(ReLU输出0，不再更新）
  - ```python
    import numpy as np
    def relu(x):
        return np.maximum(0, x)
    print(relu(-2))
    print(relu(3))
    ```
- Sigmoid(S型激活函数)
  - ![Sigmoid](https://latex.codecogs.com/svg.image?f(x)=\frac{1}{1&plus;e^{-x}})
  - 优点：能将输出压缩到（0，1），适用于二分类问题
  - 缺点：容易出现梯度消失问题
  - ```python
    import numpy as np
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    print(sigmoid(-2))
    print(sigmoid(3))
    ```
- Tanh(双曲正切)
  - ![Tanh](https://latex.codecogs.com/svg.image?f(x)=\frac{e^{x}-e^{-x}}{e^{x}&plus;e^{-x}})
  - 特点：输出范围（-1，1），比Sigmoid好，仍有可能梯度消失问题
  - ```python
    import numpy as np
    def tanh(x):
        return np.tanh(x)
    print(tanh(-2))
    print(tanh(3))
    ```
- Softmax(用于分类问题)
  - ![Softmax](https://latex.codecogs.com/svg.image?f(x_{i})=\frac{e^{x_{i}}}{\sum&space;e^{x_{j}}})
  - 适用于多分类问题
  - 把所有输出变成概率，总和为1
  - ```python
    import numpy as np
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # 防止数值溢出
        return exp_x / np.sum(exp_x)
    print(softmax([2.0, 1.0, 0.1]))
    ```

### 损失函数(Loss Function)
损失函数用于**衡量模型的预测结果与真实值的差距**，训练模型的目标是最小化损失。
#### 常见损失函数
- 均方误差(MSE, Mean Squared Error)(用于回归)
  - ![MSE](https://latex.codecogs.com/svg.image?L=\frac{1}{N}\sum(y_{true}-y_{pred})^{2})
- 交叉熵损失(Cross-Entropy Loss)(用于分类)
  - ![Cross-Entropy Loss](https://latex.codecogs.com/svg.image?L=-\sum&space;y_{true}\log(y_{pred}))

### 优化算法
优化算法用于**调整神经网络的权重**，使损失函数最小化。
#### 梯度下降(Gradient Descent)
- 计算损失函数对权重的偏导数，沿着梯度的负方向更新权重。
- ![Gradient Descent](https://latex.codecogs.com/svg.image?W=W-\alpha\frac{\partial&space;L}{\partial&space;W})
- ![](https://latex.codecogs.com/svg.image?\alpha&space;)是学习率
- ![](https://latex.codecogs.com/svg.image?\frac{\partial&space;L}{\partial&space;W})是损失函数对权重的导数
    - 随机梯度下降(SGD)
      - 每次更新仅使用一个样本，计算快但噪声较大
    - 动量梯度下降(Momentum)
      - 通过添加动量项，加速收敛
    - Adam (Adaptive Moment Estimation)
      - 结合动量和自适应学习率，是最常用的优化算法
```python
alpha = 0.01  # 学习率
gradient = np.array([0.5, -0.3, 0.1])  # 假设的梯度
W = np.random.randn(3)

W = W - alpha * gradient  # 更新权重
print("更新后的权重", W)
```

### 模型训练流程
    1. 数据准备
      - 加载数据
      - 数据预处理
    2. 模型初始化
      - 初始化权重和偏置
    3. 向前传播
      - 计算每一层的激活值
    4. 计算损失
      - 计算当前预测结果与真实值的误差
    5. 反向传播
      - 计算梯度，并调整权重
    6. 优化
      - 使用优化算法更新权重
    7. 迭代训练
      - 重复上述步骤直到损失收敛







   ## 学习路线

    ### 阶段2：工具掌握 (2-3周)
    - Python库学习
    - 基础模型实践

    ### 阶段3：核心组件 (3-4周)
    - 多层感知机
    - 损失函数
    - 优化算法
    - 模型训练流程

    ### 阶段4：进阶学习 (4-8周)
    - 图神经网络
    - Transformer架构
    - 生物信息学应用
    - 数学理论深化

    ## 模型可解释性维度

    ### 内在解释
    - 模型本身具备透明性
    - 适用于决策树、线性模型等
    - 优势在于直接可解释
    - 局限于模型复杂度

    ### 后置解释
    - 针对已训练模型
    - 使用LIME、SHAP等方法
    - 可用于复杂模型
    - 解释可靠性有限

    ### 解释范围
    - 全局解释：关注整体行为
    - 局部解释：针对单个样本
