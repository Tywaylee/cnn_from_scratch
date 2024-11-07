from .layer import *

class Network(object):
    def __init__(self):
    #     self.conv1 = Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    #     self.pool1 = MaxPooling2D()
        # Activation (ReLU)
        # self.act1 = Activation1()

        # Fully Connected Layers
        self.fc1 = FullyConnected(28 * 28, 1024)  # Fully connected after flattening
        self.bn1 = BatchNormalization(1024)
        self.act2 = Activation1()
        self.drop1 = Dropout(p = 0.1)

        # self.fc2 = FullyConnected(512, 256)
        # self.bn2 = BatchNormalization(256)
        # self.act3 = Activation1()

        self.fc3 = FullyConnected(1024, 512)  # Output layer for 10 classes
        self.bn3 = BatchNormalization(512)
        self.act4 = Activation1()
        self.drop1 = Dropout(p = 0.1)

        # self.fc4 = FullyConnected(128, 64)
        # self.bn4 = BatchNormalization(64)
        # self.act5 = Activation1()

        self.fc5 = FullyConnected(512, 10)
        self.bn5 = BatchNormalization(10)

        # Loss function
        self.softmax_loss = SoftmaxWithLoss()

    def forward(self, input, target):
        # input = input.reshape(input.shape[0], 1, 28, 28)

        # Conv2D -> ReLU
        # h1 = self.conv1.forward(input)
        # h1 = self.pool1.forward(h1)

        # Flatten the output for fully connected layers
        # h1_flattened = input.reshape(input.shape[0], -1)  # Flatten to (batch_size, 8*28*28)
        # h1 = self.act1.forward(h1_flattened)

        # Fully connected layers
        h2 = self.fc1.forward(input)
        h2 = self.bn1.forward(h2)
        h2 = self.act2.forward(h2)
        h2 = self.drop1.forward(h2)

        h4 = self.fc3.forward(h2)
        h4 = self.bn3.forward(h4)
        h4 = self.act4.forward(h4)

        h6 = self.fc5.forward(h4)
        h6 = self.bn5.forward(h6)

        # Softmax and loss
        pred, loss = self.softmax_loss.forward(h6, target)

        return pred, loss

    def backward(self):
        # Backprop through Softmax with Loss
        grad = self.softmax_loss.backward()
        grad = self.bn5.backward(grad)
        grad = self.fc5.backward(grad)

        # Backprop through fully connected layers
        # grad = self.act5.backward(grad)
        # grad = self.bn4.backward(grad)
        # grad = self.fc4.backward(grad)

        grad = self.act4.backward(grad)
        grad = self.bn3.backward(grad)
        grad = self.fc3.backward(grad)

        # grad = self.act3.backward(grad)
        # grad = self.bn2.backward(grad)
        # grad = self.fc2.backward(grad)

        grad = self.drop1.backward(grad)
        grad = self.act2.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.fc1.backward(grad)

        # Backprop through Conv2D
        # grad = grad.reshape(-1, 8, 14, 14)
        # grad = self.pool1.backward(grad)
        # grad = sgrad.reshape(-1, 8, 28, 28)
        # grad = self.conv1.backward(grad)
        


    def update(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # self.fc5.weight -= lr * self.fc5.weight_grad
        # self.fc5.bias -= lr * self.fc5.bias_grad
        # self.bn5.gamma -= lr * self.bn5.gamma_grad
        # self.bn5.beta -= lr * self.bn5.beta_grad

        # self.fc3.weight -= lr * self.fc3.weight_grad
        # self.fc3.bias -= lr * self.fc3.bias_grad
        # self.bn3.gamma -= lr * self.bn3.gamma_grad
        # self.bn3.beta -= lr * self.bn3.beta_grad

        # self.fc1.weight -= lr * self.fc1.weight_grad
        # self.fc1.bias -= lr * self.fc1.bias_grad
        # self.bn1.gamma -= lr * self.bn1.gamma_grad
        # self.bn1.beta -= lr * self.bn1.beta_grad

        # Update convolutional layer
        # self.conv1.weight -= lr * self.conv1.weight_grad
        # self.conv1.bias -= lr * self.conv1.bias_grad

        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
        # Initialize moment vectors for each parameter
        self.m_fc5_weight = np.zeros_like(self.fc5.weight)
        self.v_fc5_weight = np.zeros_like(self.fc5.weight)
        self.m_fc5_bias = np.zeros_like(self.fc5.bias)
        self.v_fc5_bias = np.zeros_like(self.fc5.bias)
        
        self.m_fc3_weight = np.zeros_like(self.fc3.weight)
        self.v_fc3_weight = np.zeros_like(self.fc3.weight)
        self.m_fc3_bias = np.zeros_like(self.fc3.bias)
        self.v_fc3_bias = np.zeros_like(self.fc3.bias)
        
        self.m_fc1_weight = np.zeros_like(self.fc1.weight)
        self.v_fc1_weight = np.zeros_like(self.fc1.weight)
        self.m_fc1_bias = np.zeros_like(self.fc1.bias)
        self.v_fc1_bias = np.zeros_like(self.fc1.bias)
        
        self.m_bn5_gamma = np.zeros_like(self.bn5.gamma)
        self.v_bn5_gamma = np.zeros_like(self.bn5.gamma)
        self.m_bn5_beta = np.zeros_like(self.bn5.beta)
        self.v_bn5_beta = np.zeros_like(self.bn5.beta)
        
        self.m_bn3_gamma = np.zeros_like(self.bn3.gamma)
        self.v_bn3_gamma = np.zeros_like(self.bn3.gamma)
        self.m_bn3_beta = np.zeros_like(self.bn3.beta)
        self.v_bn3_beta = np.zeros_like(self.bn3.beta)
        
        self.m_bn1_gamma = np.zeros_like(self.bn1.gamma)
        self.v_bn1_gamma = np.zeros_like(self.bn1.gamma)
        self.m_bn1_beta = np.zeros_like(self.bn1.beta)
        self.v_bn1_beta = np.zeros_like(self.bn1.beta)
        
        self.t += 1
        
        # Update for fc5 weight and bias
        self.m_fc5_weight = self.beta1 * self.m_fc5_weight + (1 - self.beta1) * self.fc5.weight_grad
        self.v_fc5_weight = self.beta2 * self.v_fc5_weight + (1 - self.beta2) * (self.fc5.weight_grad ** 2)
        m_hat_fc5_weight = self.m_fc5_weight / (1 - self.beta1 ** self.t)
        v_hat_fc5_weight = self.v_fc5_weight / (1 - self.beta2 ** self.t)
        self.fc5.weight -= self.learning_rate * m_hat_fc5_weight / (np.sqrt(v_hat_fc5_weight) + self.epsilon)
        
        self.m_fc5_bias = self.beta1 * self.m_fc5_bias + (1 - self.beta1) * self.fc5.bias_grad
        self.v_fc5_bias = self.beta2 * self.v_fc5_bias + (1 - self.beta2) * (self.fc5.bias_grad ** 2)
        m_hat_fc5_bias = self.m_fc5_bias / (1 - self.beta1 ** self.t)
        v_hat_fc5_bias = self.v_fc5_bias / (1 - self.beta2 ** self.t)
        self.fc5.bias -= self.learning_rate * m_hat_fc5_bias / (np.sqrt(v_hat_fc5_bias) + self.epsilon)
        
        # Update for bn5 gamma and beta
        self.m_bn5_gamma = self.beta1 * self.m_bn5_gamma + (1 - self.beta1) * self.bn5.gamma_grad
        self.v_bn5_gamma = self.beta2 * self.v_bn5_gamma + (1 - self.beta2) * (self.bn5.gamma_grad ** 2)
        m_hat_bn5_gamma = self.m_bn5_gamma / (1 - self.beta1 ** self.t)
        v_hat_bn5_gamma = self.v_bn5_gamma / (1 - self.beta2 ** self.t)
        self.bn5.gamma -= self.learning_rate * m_hat_bn5_gamma / (np.sqrt(v_hat_bn5_gamma) + self.epsilon)
        
        self.m_bn5_beta = self.beta1 * self.m_bn5_beta + (1 - self.beta1) * self.bn5.beta_grad
        self.v_bn5_beta = self.beta2 * self.v_bn5_beta + (1 - self.beta2) * (self.bn5.beta_grad ** 2)
        m_hat_bn5_beta = self.m_bn5_beta / (1 - self.beta1 ** self.t)
        v_hat_bn5_beta = self.v_bn5_beta / (1 - self.beta2 ** self.t)
        self.bn5.beta -= self.learning_rate * m_hat_bn5_beta / (np.sqrt(v_hat_bn5_beta) + self.epsilon)
        
        # Update for fc3 weight and bias
        self.m_fc3_weight = self.beta1 * self.m_fc3_weight + (1 - self.beta1) * self.fc3.weight_grad
        self.v_fc3_weight = self.beta2 * self.v_fc3_weight + (1 - self.beta2) * (self.fc3.weight_grad ** 2)
        m_hat_fc3_weight = self.m_fc3_weight / (1 - self.beta1 ** self.t)
        v_hat_fc3_weight = self.v_fc3_weight / (1 - self.beta2 ** self.t)
        self.fc3.weight -= self.learning_rate * m_hat_fc3_weight / (np.sqrt(v_hat_fc3_weight) + self.epsilon)
        
        self.m_fc3_bias = self.beta1 * self.m_fc3_bias + (1 - self.beta1) * self.fc3.bias_grad
        self.v_fc3_bias = self.beta2 * self.v_fc3_bias + (1 - self.beta2) * (self.fc3.bias_grad ** 2)
        m_hat_fc3_bias = self.m_fc3_bias / (1 - self.beta1 ** self.t)
        v_hat_fc3_bias = self.v_fc3_bias / (1 - self.beta2 ** self.t)
        self.fc3.bias -= self.learning_rate * m_hat_fc3_bias / (np.sqrt(v_hat_fc3_bias) + self.epsilon)
        
        # Update for bn3 gamma and beta
        self.m_bn3_gamma = self.beta1 * self.m_bn3_gamma + (1 - self.beta1) * self.bn3.gamma_grad
        self.v_bn3_gamma = self.beta2 * self.v_bn3_gamma + (1 - self.beta2) * (self.bn3.gamma_grad ** 2)
        m_hat_bn3_gamma = self.m_bn3_gamma / (1 - self.beta1 ** self.t)
        v_hat_bn3_gamma = self.v_bn3_gamma / (1 - self.beta2 ** self.t)
        self.bn3.gamma -= self.learning_rate * m_hat_bn3_gamma / (np.sqrt(v_hat_bn3_gamma) + self.epsilon)
        
        self.m_bn3_beta = self.beta1 * self.m_bn3_beta + (1 - self.beta1) * self.bn3.beta_grad
        self.v_bn3_beta = self.beta2 * self.v_bn3_beta + (1 - self.beta2) * (self.bn3.beta_grad ** 2)
        m_hat_bn3_beta = self.m_bn3_beta / (1 - self.beta1 ** self.t)
        v_hat_bn3_beta = self.v_bn3_beta / (1 - self.beta2 ** self.t)
        self.bn3.beta -= self.learning_rate * m_hat_bn3_beta / (np.sqrt(v_hat_bn3_beta) + self.epsilon)
        
        # Update for fc1 weight and bias
        self.m_fc1_weight = self.beta1 * self.m_fc1_weight + (1 - self.beta1) * self.fc1.weight_grad
        self.v_fc1_weight = self.beta2 * self.v_fc1_weight + (1 - self.beta2) * (self.fc1.weight_grad ** 2)
        m_hat_fc1_weight = self.m_fc1_weight / (1 - self.beta1 ** self.t)
        v_hat_fc1_weight = self.v_fc1_weight / (1 - self.beta2 ** self.t)
        self.fc1.weight -= self.learning_rate * m_hat_fc1_weight / (np.sqrt(v_hat_fc1_weight) + self.epsilon)
        
        self.m_fc1_bias = self.beta1 * self.m_fc1_bias + (1 - self.beta1) * self.fc1.bias_grad
        self.v_fc1_bias = self.beta2 * self.v_fc1_bias + (1 - self.beta2) * (self.fc1.bias_grad ** 2)
        m_hat_fc1_bias = self.m_fc1_bias / (1 - self.beta1 ** self.t)
        v_hat_fc1_bias = self.v_fc1_bias / (1 - self.beta2 ** self.t)
        self.fc1.bias -= self.learning_rate * m_hat_fc1_bias / (np.sqrt(v_hat_fc1_bias) + self.epsilon)
        
        # Update for bn1 gamma and beta
        self.m_bn1_gamma = self.beta1 * self.m_bn1_gamma + (1 - self.beta1) * self.bn1.gamma_grad
        self.v_bn1_gamma = self.beta2 * self.v_bn1_gamma + (1 - self.beta2) * (self.bn1.gamma_grad ** 2)
        m_hat_bn1_gamma = self.m_bn1_gamma / (1 - self.beta1 ** self.t)
        v_hat_bn1_gamma = self.v_bn1_gamma / (1 - self.beta2 ** self.t)
        self.bn1.gamma -= self.learning_rate * m_hat_bn1_gamma / (np.sqrt(v_hat_bn1_gamma) + self.epsilon)
        
        self.m_bn1_beta = self.beta1 * self.m_bn1_beta + (1 - self.beta1) * self.bn1.beta_grad
        self.v_bn1_beta = self.beta2 * self.v_bn1_beta + (1 - self.beta2) * (self.bn1.beta_grad ** 2)
        m_hat_bn1_beta = self.m_bn1_beta / (1 - self.beta1 ** self.t)
        v_hat_bn1_beta = self.v_bn1_beta / (1 - self.beta2 ** self.t)
        self.bn1.beta -= self.learning_rate * m_hat_bn1_beta / (np.sqrt(v_hat_bn1_beta) + self.epsilon)
        
