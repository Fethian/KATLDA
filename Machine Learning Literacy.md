# 常见的超参数及其作用

## 学习率（Learning Rate）
- 控制每次参数更新的步长
- 学习率过大可能导致模型在最优解附近震荡
- 过小则可能导致收敛速度过慢

## 批量大小（Batch Size）
- 每次迭代中用于训练的样本数量
- 较大的批量大小可以提高训练效率，但可能使模型陷入局部最优解
- 较小的批量大小可以帮助模型更好地泛化，但可能增加训练时间

## 迭代次数（Epochs）
- 训练过程中完整遍历训练数据集的次数
- 更多的迭代次数可以使模型学习更充分，但过多可能导致过拟合

## 网络结构相关超参数
- 包括网络的层数
- 每层的神经元数量
- 激活函数的选择等
- 这些超参数直接影响模型的表达能力和复杂度

## 优化器参数
- 动量（Momentum）
- 权重衰减（Weight Decay）等
- 这些参数影响参数更新的方式和速度

## 正则化参数
- 用于控制模型的复杂度，防止过拟合
- 常见的正则化方法包括L1正则化、L2正则化等

## Dropout率
- 在训练过程中随机丢弃神经元的比例
- 有助于减少过拟合