import numpy as np

'''
    Convolutional Neural Network implemented from scratch using numpy

    Process:
        1. Input the data into the neural network
        2. Data flows from one layer to another layer until we get the output
        3. Compute the loss, a scalar
        4. Backpropagate the loss to each layer *(Most Important)
        5. Update the weights and biases using the gradients
        6. Iterate until the loss is minimized

    How to build every layer?
        X -> Layer -> Y -> Loss -> Gradients -> Weights and Biases -> Update


    For more info, contact me at:
    orionisfan@outlook.com


'''

def conv_forward(input, weights, bias, stride=1, padding=0):
    '''
    Input:
        input: 4D numpy array, shape (batch_size, in_channels, in_height, in_width)
        weights: 4D numpy array, shape (out_channels, in_channels, kernel_height, kernel_width)
        bias: 1D numpy array, shape (out_channels,)
        stride: int, default 1
        padding: int, default 0
    '''

    pass
    return 0





'''
虽然可以使用Numpy从头开始搭建卷积神经网络(CNN),但这需要大量的代码实现,且计算效率不如深度学习框架如PyTorch和TensorFlow。通常我们使用Numpy进行数据处理,而用专门的深度学习库搭建和训练CNN。下面我简要介绍用Numpy实现CNN的基本思路:

1. 定义卷积层的前向传播函数:
```python
def conv_forward(input, weights, bias, stride=1, padding=0):
    # 根据stride和padding对输入进行填充和卷积计算
    # 卷积的实现可以通过im2col函数将输入展开,再用矩阵乘法实现
    pass
```

2. 定义ReLU激活函数:  
```python
def relu_forward(input):
    return np.maximum(0, input)
```

3. 定义池化层的前向传播函数,以最大池化为例:
```python
def maxpool_forward(input, pool_size=2, stride=2):
    # 根据pool_size和stride对输入进行最大池化操作
    pass
```

4. 定义全连接层的前向传播函数:
```python
def fc_forward(input, weights, bias):
    return np.dot(input, weights) + bias
```

5. 定义Softmax损失函数:
```python
def softmax_loss(scores, labels):
    # 根据分类scores计算交叉熵损失
    pass
```

6. 定义卷积、池化、全连接层的反向传播函数,用于计算梯度:
```python
def conv_backward(dout, cache):
    pass

def relu_backward(dout, cache):  
    pass

def maxpool_backward(dout, cache):
    pass

def fc_backward(dout, cache):
    pass
```

7. 组合以上层构建CNN模型的前向传播:
```python
def forward(x, conv_w, conv_b, fc_w, fc_b):
    conv_out = conv_forward(x, conv_w, conv_b)
    relu_out = relu_forward(conv_out)
    pool_out = maxpool_forward(relu_out)
    fc_out = fc_forward(pool_out, fc_w, fc_b)
    return fc_out
```

8. 使用随机梯度下降等优化算法,不断迭代前向传播和反向传播过程,训练CNN参数直至收敛。

以上是用Numpy手工实现CNN的基本流程和思路,可以帮助我们理解CNN的内部原理。但实践中,还是建议直接使用成熟的深度学习框架如Keras、PyTorch搭建CNN,可以极大提高开发效率和训练速度。这些高层框架已经对卷积、池化等操作进行了优化和封装,且能自动求导,不需要手工推导和编写反向传播的代码。

'''