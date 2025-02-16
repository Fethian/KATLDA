【机器学习】：任务T  性能度量方案P/ 计算机程序：自主学习任务T的经验E
    # 机器学习基础知识

    ## 监督学习 (Supervised Learning, SL)
    使用标记输入输出已知的样本数据。主要解决两类问题：回归(regression)和分类(classification)

    ### 常用算法
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

    ## 无监督学习 (Unsupervised Learning, UL)
    使用未标记的样本数据，在未知输入和预先确定的输出之间建立连接。主要解决：输入数据聚类(Clustering)和输入特征变量关联(Correlation)。

    ### 常用算法
    - K均值聚类 (K-Means Clustering)
    - DBSCAN (Density-based Spatial Clustering of Applications with Noise)
    - 主成分分析算法 (Principal Component Analysis, PCA)
    - 自组织映射神经网络 (Self-Organizing Map, SOM)
    - 受限玻尔兹曼机 (Restricted Boltzmann Machine, RBM)

    ## 强化学习 (Reinforcement Learning, RL)
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

    ## 学习路线

    ### 阶段1：基础概念 (1-2周)
    - 机器学习基本类型理解
    - 推荐资源：
        - 《Hands-On Machine Learning》
        - Andrew Ng的Coursera课程

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
