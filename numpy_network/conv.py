import numpy as np

input = np.array([1, 2, 3, 4, 5])
kernel = np.array([0.5, 1, 0.5])

'''First method: using numpy.convolve() function
    
    This method is suitable for 1D, 2D and 3D convolution.
    
    The function numpy.convolve() computes the convolution of two one-dimensional sequences.
    
    It takes two arguments: the input sequence and the kernel, and returns the output sequence.
    
    The kernel is a one-dimensional sequence that is convolved with the input sequence.
    
    The output sequence has the same length as the input sequence, and represents the result of the convolution.
    
    The convolution is computed by multiplying the input sequence with the kernel, and then summing up the products.
    
    The result is a single value for each position in the output sequence.
    The convolution can be performed using the following code:

'''
def conv_anyd(input, kernel):
        '''
        This function performs convolution on any-dimensional input.
        '''
        return np.convolve(input, kernel, mode='valid')

'''Second method: implementing convolution without using numpy.convolve() function
    This method is suitable for 1D convolution only.
'''
def conv1d_handler(input, kernel, mode):
        if mode == 0 or mode == 'same':
            convolution_result = np.zeros(input.shape[0])
            
            if kernel.shape[0] % 2 == 0:
                left_padding = int(kernel.shape[0] / 2)
                right_padding = left_padding - 1
            else:
                padding = int((kernel.shape[0] - 1) / 2)

            for i in range(convolution_result.shape[0]):
                pass


        elif mode == 1 or mode == 'valid':
            convolution_result = np.zeros(input.shape[0] - kernel.shape[0] + 1)
            for i in range(convolution_result.shape[0]):
                convolution_result[i] = np.sum(input[i:i+kernel.shape[0]] * kernel)
        
        else:
            raise ValueError("Unsupported mode, only''same'' or '0' and ''valid'' or '1' are supported.")

        return convolution_result
output = conv1d_handler(input, kernel, 0)
print(output)


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