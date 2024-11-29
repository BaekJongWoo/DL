import numpy as np
from mine.model import ModelBase
from mine.activation import tanh, sigmoid

class RecurentBase(ModelBase): # many to one base
    def __init__(self, to_many):
        super().__init__()
        self.to_many = to_many

    def stepForward(self, x: np.ndarray, h) -> np.ndarray:
        raise NotImplementedError()

    def stepBackward(self, x: np.ndarray, h: np.ndarray, y:np.ndarray, grad_y: np.ndarray, grad_h: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.xt = []
        self.ht = []
        self.yt = []

        self.n = x.shape[1]
        h = None
        
        self.seq_lengths = np.sum(np.any(x != 0, axis=2), axis=1)
        print(self.seq_lengths)

        for i in range(self.n):
            x_slice = x[:, i, :]
            h_next, y = self.stepForward(x_slice, h)
            
            self.xt.append(x_slice)
            self.ht.append(h if h is not None else np.zeros_like(h_next))
            self.yt.append(y)

            h = h_next
    
        batch_size = x.shape[0]

        if self.to_many:
            outputs = np.stack(self.yt, axis=1)

            for b in range(batch_size):
                valid_length = self.seq_lengths[b]
                outputs[b, valid_length:, :] = 0

            return outputs
        else:
            final_outputs = np.zeros((batch_size, y.shape[-1]))

            for b in range(batch_size):
                valid_length = self.seq_lengths[b] - 1
                final_outputs[b] = self.yt[valid_length][b]

            return final_outputs
    
    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        grad_xt = []

        # Mask the gradients to ignore padded time steps
        for b in range(grad_y.shape[0]):
            valid_length = self.seq_lengths[b]  # Use seq_lengths calculated during forward
            grad_y[b, valid_length:, :] = 0  # Set gradients of padded time steps to 0

        grad_h = None  # Initialize hidden state gradient
        for t in reversed(range(self.n)):
            x = self.xt.pop()
            h = self.ht.pop()
            y = self.yt.pop()

            # Apply backward step for current time step
            grad_x, grad_h = self.stepBackward(x, h, y, grad_y[:, t, :], grad_h)
            grad_xt.insert(0, grad_x)

        return np.stack(grad_xt, axis=1)

class LSTM(RecurentBase):
    def __init__(self, x_size: int, h_size: int, to_many: bool):
        super().__init__(to_many)
        self.h_size = h_size
        self.Wf = np.random.uniform(-1, 1, (x_size + h_size, h_size))
        self.Wi = np.random.uniform(-1, 1, (x_size + h_size, h_size))
        self.Wo = np.random.uniform(-1, 1, (x_size + h_size, h_size))
        self.Wc = np.random.uniform(-1, 1, (x_size + h_size, h_size))

        self.bf = np.random.uniform(-1, 1, (1, h_size))
        self.bi = np.random.uniform(-1, 1, (1, h_size))
        self.bo = np.random.uniform(-1, 1, (1, h_size))
        self.bc = np.random.uniform(-1, 1, (1, h_size))

    def stepForward(self, x, h):
        if h is None:
            h_prev = np.zeros((x.shape[0], self.h_size))
            c_prev = np.zeros((x.shape[0], self.h_size))
        else:
            h_prev, c_prev = h

        xh = np.concatenate([x,h_prev], axis=1)
        ft = sigmoid.s_forward(np.dot(xh, self.Wf) + self.bf)
        it = sigmoid.s_forward(np.dot(xh, self.Wi) + self.bi)
        ct = sigmoid.s_forward(np.dot(xh, self.Wc) + self.bc)
        ot = sigmoid.s_forward(np.dot(xh, self.Wo) + self.bo)

        c_next = ft * c_prev + it * ct
        h_next = tanh.s_forward(c_next) * ot

        return (c_next, h_next), h_next
    
    def stepBackward(self, x, h, y, grad_y, grad_h):
        h_prev, c_prev = h
        if grad_h is None:
            grad_h = np.zeros((x.shape[0], self.h_size))
            grad_c = np.zeros((x.shape[0], self.h_size))
        else:
            grad_h, grad_c = grad_h 
                


class RNN(RecurentBase):
    def __init__(self, x_size: int, h_size: int):
        super().__init__()
        self.Wx = np.random.uniform(-1, 1, (x_size, h_size))
        self.Wh = np.random.uniform(-1, 1, (h_size, h_size))
        self.b = np.random.uniform(-1, 1, (1, h_size))

    def stepForward(self, x: np.ndarray, h: np.ndarray):
        if h is None:
            h = np.zeros((x.shape[0], self.Wh.shape[0]))

        a = np.dot(x, self.Wx) + np.dot(h, self.Wh) + self.b
        h = tanh.s_forward(a)
        return h, h

    def stepBackward(self, x: np.ndarray, h: np.ndarray, y:np.ndarray, grad_y: np.ndarray, grad_h: np.ndarray) -> np.ndarray:
        batch_size = grad_y.shape[0]
        if grad_h is None:
            grad_h = np.zeros((x.shape[0], self.Wh.shape[0]))

        grad = grad_y + grad_h
        grad = tanh.s_backward(y, grad)
        
        dx = np.dot(grad, self.Wx.T)
        dWx = np.dot(x.T, grad) / batch_size
        dh = np.dot(grad, self.Wh.T)
        dWh = np.dot(h.T, grad) / batch_size
        db = np.sum(grad, axis=0, keepdims=True) / batch_size

        self.optimizer.step([self.Wx, self.Wh, self.b], [dWx, dWh, db])

        return dx, dh

class Linear(ModelBase):
    def __init__(self, input_size:int, output_size:int) -> None:
        self.W = np.random.uniform(-1, 1, (input_size, output_size))
        self.b = np.random.uniform(-1, 1, (1, output_size))

        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch_size = grad.shape[0]

        dx = np.dot(grad, self.W.T)
        dW = np.dot(self.x.T, grad) / batch_size
        db = np.sum(grad, axis=0, keepdims=True) / batch_size

        self.optimizer.step([self.W, self.b], [dW, db])

        return dx

class Conv2D(ModelBase):
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

        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride
        
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