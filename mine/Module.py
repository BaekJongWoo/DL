import numpy as np

class Module():
    def forward(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def backward(self, grad:np.ndarray, learning_rate:float) -> np.ndarray:
        raise NotImplementedError()

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        return x.reshape(self.input_shape[0], -1)

    def backward(self, grad: np.ndarray, learning_rate:float) -> np.ndarray:
        return grad.reshape(self.input_shape)

class Linear(Module):
    def __init__(self, input_size:int, output_size:int) -> None:
        self.W = np.random.uniform(-1, 1, (input_size, output_size))
        self.b = np.random.uniform(-1, 1, (1, output_size))

        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size = grad.shape[0]

        dx = np.dot(grad, self.W.T)
        dW = np.dot(self.x.T, grad) / batch_size
        db = np.sum(grad, axis=0, keepdims=True) / batch_size

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dx

class Conv2D(Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = np.random.uniform(-1, 1, (output_channels, input_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-1, 1, (1, output_channels))

        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        batch_size, in_channels, in_height, in_width = x.shape

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        out = np.zeros((batch_size, self.output_channels, out_height, out_width))

        for yi in range(out_height):
            for xi in range(out_width):
                h_start = yi * self.stride
                h_end = h_start + self.kernel_size
                w_start = xi * self.stride
                w_end = w_start + self.kernel_size

                patch = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, yi, xi] = np.tensordot(patch, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b

        return out

    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size, out_channels, out_height, out_width = grad.shape

        if self.padding > 0:
            x_padded = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = self.x

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dx_padded = np.zeros_like(x_padded)

        for yi in range(out_height):
            for xi in range(out_width):
                h_start = yi * self.stride
                h_end = h_start + self.kernel_size
                w_start = xi * self.stride
                w_end = w_start + self.kernel_size

                grad_value = grad[:, :, yi, xi]
                patch = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                dW += np.tensordot(grad_value, patch, axes=([0], [0]))
                db += np.sum(grad_value, axis=(0))
                dx_padded[:, :, h_start:h_end, w_start:w_end] += np.tensordot(grad_value, self.W, axes=([1], [0]))

        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded

        self.W -= learning_rate * dW / batch_size
        self.b -= learning_rate * db / batch_size

        return dx

class MaxPooling(Module):
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.argmax = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        batch_size, channels, height, width = x.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))
        self.argmax = np.zeros_like(x, dtype=int)

        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size

                pooling_region = x[:, :, h_start:h_end, w_start:w_end]
                max_value = np.max(pooling_region, axis=(2, 3))
                output[:, :, h, w] = max_value

                max_index = np.argmax(pooling_region.reshape(batch_size, channels, -1), axis=2)
                max_pos = np.unravel_index(max_index, (self.pool_size, self.pool_size))

                max_pos_row = max_pos[0] + h_start
                max_pos_col = max_pos[1] + w_start
                self.argmax[np.arange(batch_size)[:, None], np.arange(channels), max_pos_row, max_pos_col] = 1

        return output

    def backward(self, grad: np.ndarray, learning_rate:float) -> np.ndarray:
        batch_size, channels, out_height, out_width = grad.shape
        dx = np.zeros_like(self.x)

        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size

                grad_slice = grad[:, :, h, w].reshape(batch_size, channels, 1, 1)
                dx[:, :, h_start:h_end, w_start:w_end] += grad_slice * self.argmax[:, :, h_start:h_end, w_start:w_end]

        return dx

class BatchNorm1D(Module):
    def __init__(self, epsilon=1e-5) -> None:
        self.epsilon = epsilon

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_mean = np.mean(x, axis=0, keepdims=True)
        self.batch_var = np.var(x, axis=0, keepdims=True)
        self.x_normalized = (x - batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        return self.x_normalized
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        N, D = grad.shape
        
        x_mu = self.x_normalized * np.sqrt(self.batch_var + self.epsilon)
        inv_var = 1.0 / np.sqrt(self.batch_var + self.epsilon)

        dx_normalized = grad
        dvar = np.sum(dx_normalized * (x_mu) * (-0.5) * inv_var**3, axis=0)
        dmean = np.sum(dx_normalized * (-inv_var), axis=0) + dvar * np.mean(-2.0 * x_mu, axis=0)        
        dx = (dx_normalized * inv_var) + (dvar * 2.0 * x_mu / N) + (dmean / N)
        
        return dx

class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        return (self.x > 0) * grad