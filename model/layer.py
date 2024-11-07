import numpy as np

class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))


    def forward(self, input):
        self.input = input
        output = np.matmul(input, self.weight)
        output = np.add(output, self.bias)

        return output

    def backward(self, output_grad):
        input_grad = np.matmul(output_grad, self.weight.T)
        self.weight_grad = np.matmul(self.input.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis = 0, keepdims = True)

        return input_grad

class Activation1(_Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)

        return output

    def backward(self, output_grad):
        input_grad = output_grad * (self.input > 0)

        return input_grad

class SoftmaxWithLoss(_Layer):
    def __init__(self):
        pass

    def forward(self, input, target):
        self.input = input
        self.target = target

        '''Softmax'''
        shifted_input = input - np.max(input, axis=1, keepdims=True)
        exp_input = np.exp(shifted_input)
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)

        '''Average loss'''
        if len(target.shape) == 1:
            target_one_hot = np.zeros_like(self.output)
            target_one_hot[np.arange(target.shape[0]), target] = 1
        else:
            target_one_hot = target
        
        # Compute the cross-entropy loss (average loss over batch)
        loss = -np.sum(target_one_hot * np.log(self.output + 1e-7)) / input.shape[0] 

        return self.output, loss

    def backward(self):
        batch_size = self.input.shape[0]

        # If target is class indices (not one-hot), convert to one-hot
        if len(self.target.shape) == 1:
            target_one_hot = np.zeros_like(self.output)
            target_one_hot[np.arange(self.target.shape[0]), self.target] = 1
        else:
            target_one_hot = self.target

        # Gradient of cross-entropy loss with respect to the input logits
        input_grad = (self.output - target_one_hot) / batch_size
        
        return input_grad

class Conv2D(_Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize the Conv2D layer.
        
        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        :param out_channels: Number of output channels (number of filters).
        :param kernel_size: Size of the convolutional kernel/filter (assumed to be square).
        :param stride: Step size for moving the kernel.
        :param padding: Number of zero-padding rows/columns added to the input.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros((out_channels, 1))

        # To store the gradients for backward pass
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, input):
        """
        Perform the forward pass for Conv2D.
        
        :param input: Input image(s), shape (batch_size, in_channels, height, width)
        :return: Output feature maps, shape (batch_size, out_channels, out_height, out_width)
        """
        self.input = input
        batch_size, in_channels, in_height, in_width = input.shape
        
        # Calculate output dimensions
        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        # Apply padding to the input if necessary
        if self.padding > 0:
            input_padded = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            input_padded = input
        
        # Initialize the output feature map
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                # Define the current region of interest in the input
                region = input_padded[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                # Perform convolution: sum over element-wise multiplication between region and weights
                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(region * self.weight[k, :, :, :], axis=(1, 2, 3)) + self.bias[k]
        
        return output

    def backward(self, output_grad):
        """
        Perform the backward pass for Conv2D.
        
        :param output_grad: Gradient of the loss with respect to the output, shape (batch_size, out_channels, out_height, out_width)
        :return: Gradient of the loss with respect to the input (input_grad)
        """
        batch_size, in_channels, in_height, in_width = self.input.shape
        out_height, out_width = output_grad.shape[2], output_grad.shape[3]
        
        # Apply padding to the input if necessary
        if self.padding > 0:
            input_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            input_grad_padded = np.zeros_like(input_padded)
        else:
            input_padded = self.input
            input_grad_padded = np.zeros_like(self.input)
        
        # Initialize gradients
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)

        # Compute gradients
        for i in range(out_height):
            for j in range(out_width):
                # Define the current region of interest in the input
                region = input_padded[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                
                # Compute gradient for weights and bias
                for k in range(self.out_channels):
                    self.weight_grad[k, :, :, :] += np.sum(region * (output_grad[:, k, i, j])[:, None, None, None], axis=0)
                    self.bias_grad[k] += np.sum(output_grad[:, k, i, j], axis=0)
                    
                # Compute gradient for the input
                for b in range(batch_size):
                    for k in range(self.out_channels):
                        input_grad_padded[b, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size] += self.weight[k, :, :, :] * output_grad[b, k, i, j]
        
        # Remove padding from the input gradient if padding was applied
        if self.padding > 0:
            input_grad = input_grad_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_grad = input_grad_padded

        return input_grad

class BatchNormalization(_Layer):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        """
        Initialize Batch Normalization layer.
        
        :param num_features: The number of features in the input.
        :param momentum: Momentum for the running mean and variance.
        :param epsilon: Small value to avoid division by zero.
        """
        self.gamma = np.ones((1, num_features))  # Scaling parameter (learnable)
        self.beta = np.zeros((1, num_features))  # Shifting parameter (learnable)
        self.momentum = momentum
        self.epsilon = epsilon

        # Running statistics for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        # Store intermediate results for backward pass
        self.input_normalized = None
        self.mean = None
        self.var = None
        self.input = None

    def forward(self, input, is_training=True):
        """
        Forward pass for Batch Normalization.
        
        :param input: Input data, shape (batch_size, num_features)
        :param is_training: Boolean flag for training or inference.
        """
        self.input = input

        if is_training:
            # Compute mean and variance over the batch
            self.mean = np.mean(input, axis=0, keepdims=True)
            self.var = np.var(input, axis=0, keepdims=True)

            # Normalize the input
            self.input_normalized = (input - self.mean) / np.sqrt(self.var + self.epsilon)

            # Update running statistics (for inference)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            # Use running statistics for inference
            self.input_normalized = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Apply scale (gamma) and shift (beta)
        output = self.gamma * self.input_normalized + self.beta
        return output

    def backward(self, output_grad):
        """
        Backward pass for Batch Normalization.
        
        :param output_grad: Gradient of the loss with respect to the output of this layer.
        :return: Gradient of the loss with respect to the input.
        """
        batch_size = output_grad.shape[0]

        # Gradients for gamma and beta
        gamma_grad = np.sum(output_grad * self.input_normalized, axis=0, keepdims=True)
        beta_grad = np.sum(output_grad, axis=0, keepdims=True)

        # Gradient of the loss with respect to the normalized input
        input_normalized_grad = output_grad * self.gamma

        # Gradient of the loss with respect to variance
        var_grad = np.sum(input_normalized_grad * (self.input - self.mean) * -0.5 * (self.var + self.epsilon)**(-1.5), axis=0, keepdims=True)

        # Gradient of the loss with respect to mean
        mean_grad = np.sum(input_normalized_grad * -1 / np.sqrt(self.var + self.epsilon), axis=0, keepdims=True) + \
                    var_grad * np.sum(-2 * (self.input - self.mean), axis=0, keepdims=True) / batch_size

        # Gradient of the loss with respect to the input
        input_grad = input_normalized_grad * 1 / np.sqrt(self.var + self.epsilon) + \
                     var_grad * 2 * (self.input - self.mean) / batch_size + \
                     mean_grad / batch_size

        # Save gradients for gamma and beta to update them during optimization
        self.gamma_grad = gamma_grad
        self.beta_grad = beta_grad

        return input_grad

class MaxPooling2D(_Layer):
    def __init__(self, pool_size=2, stride=2):
        """
        Initialize the MaxPooling2D layer.
        
        :param pool_size: The size of the pooling window (assumed to be square).
        :param stride: The step size for moving the pooling window.
        """
        self.pool_size = pool_size
        self.stride = stride

        # Store indices of max values for backward pass
        self.input = None
        self.max_indices = None

    def forward(self, input):
        """
        Forward pass for MaxPooling2D.
        
        :param input: Input data, shape (batch_size, in_channels, height, width)
        :return: Output after max pooling, shape (batch_size, in_channels, out_height, out_width)
        """
        self.input = input
        batch_size, in_channels, in_height, in_width = input.shape

        # Calculate output dimensions
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1

        # Initialize output and store max indices
        output = np.zeros((batch_size, in_channels, out_height, out_width))
        self.max_indices = np.zeros_like(input, dtype=bool)

        # Perform max pooling
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Get the current pooling region
                        start_i = i * self.stride
                        start_j = j * self.stride
                        pool_region = input[b, c, start_i:start_i + self.pool_size, start_j:start_j + self.pool_size]

                        # Find the max value and its index
                        max_value = np.max(pool_region)
                        max_index = np.unravel_index(np.argmax(pool_region), pool_region.shape)

                        # Set the output and store the index of the max value
                        output[b, c, i, j] = max_value
                        self.max_indices[b, c, start_i + max_index[0], start_j + max_index[1]] = True

        return output

    def backward(self, output_grad):
        """
        Backward pass for MaxPooling2D.
        
        :param output_grad: Gradient of the loss with respect to the output of this layer.
        :return: Gradient of the loss with respect to the input of this layer.
        """
        batch_size, in_channels, in_height, in_width = self.input.shape
        input_grad = np.zeros_like(self.input)

        # Distribute the gradients back to the positions of the max values
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(output_grad.shape[2]):
                    for j in range(output_grad.shape[3]):
                        start_i = i * self.stride
                        start_j = j * self.stride

                        # Only pass the gradient to the max value's position
                        input_grad[b, c, start_i:start_i + self.pool_size, start_j:start_j + self.pool_size][self.max_indices[b, c, start_i:start_i + self.pool_size, start_j:start_j + self.pool_size]] = output_grad[b, c, i, j]

        return input_grad


class Dropout(_Layer):
    def __init__(self, p=0.5):
        """
        Initialize Dropout layer.
        
        :param p: Dropout probability. The probability of dropping a unit. Default is 0.5.
        """
        self.p = p
        self.mask = None

    def forward(self, input, is_training=True):
        """
        Forward pass for Dropout.
        
        :param input: Input data, shape (batch_size, num_features)
        :param is_training: Boolean flag for training or inference.
        :return: Output after applying dropout, same shape as input.
        """
        if is_training:
            # Create a mask for dropout
            self.mask = (np.random.rand(*input.shape) > self.p) / (1 - self.p)
            output = input * self.mask
        else:
            # During inference, we do nothing (just pass the input through)
            output = input
        
        return output

    def backward(self, output_grad):
        """
        Backward pass for Dropout.
        
        :param output_grad: Gradient of the loss with respect to the output of this layer.
        :return: Gradient of the loss with respect to the input.
        """
        # Apply the dropout mask to the gradient
        input_grad = output_grad * self.mask
        return input_grad
