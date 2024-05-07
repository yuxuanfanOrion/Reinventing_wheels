import numpy as np

class ConvolutionLayer:
    def __init__(self, weight, stride=1):
        """
        :param weight: 卷积核，可以是一维、二维或三维数组。
        :param stride: 卷积步长，默认为1。
        """
        self.weight = np.array(weight)
        self.stride = stride

    def forward(self, input):
        """
        根据输入的维度执行卷积操作。
        
        :param input: 输入数据，可以是一维、二维或三维数组。
        :return: 卷积结果。
        """
        if input.ndim == 1:
            return self._conv1d(input)
        elif input.ndim == 2:
            return self._conv2d(input)
        elif input.ndim == 3:
            return self._conv3d(input)
        else:
            raise ValueError("Unsupported input dimensions")

    def _conv1d(self, x):
        output_length = (x.shape[0] - self.weight.shape[0]) // self.stride + 1
        output = np.zeros(output_length)
        for i in range(output_length):
            output[i] = np.sum(x[i*self.stride:i*self.stride+self.weight.shape[0]] * self.weight)
        return output

    def _conv2d(self, x):
        output_height = (x.shape[0] - self.weight.shape[0]) // self.stride + 1
        output_width = (x.shape[1] - self.weight.shape[1]) // self.stride + 1
        output = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                output[i, j] = np.sum(x[i*self.stride:i*self.stride+self.weight.shape[0],
                                          j*self.stride:j*self.stride+self.weight.shape[1]] * self.weight)
        return output

    def _conv3d(self, x):
        output_depth = (x.shape[0] - self.weight.shape[0]) // self.stride + 1
        output_height = (x.shape[1] - self.weight.shape[1]) // self.stride + 1
        output_width = (x.shape[2] - self.weight.shape[2]) // self.stride + 1
        output = np.zeros((output_depth, output_height, output_width))
        for i in range(output_depth):
            for j in range(output_height):
                for k in range(output_width):
                    output[i, j, k] = np.sum(x[i*self.stride:i*self.stride+self.weight.shape[0],
                                                j*self.stride:j*self.stride+self.weight.shape[1],
                                                k*self.stride:k*self.stride+self.weight.shape[2]] * self.weight)
        return output