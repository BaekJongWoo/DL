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

        dx = np.dot(grad, self.W.T)  # (batch_size, input_size)
        dW = np.dot(self.x.T, grad) / batch_size  # (input_size, output_size)
        db = np.sum(grad, axis=0, keepdims=True) / batch_size  # (1, output_size)

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
        self.b = np.random.uniform(-1, 1, (output_channels, 1))

        self.x = None  # 입력 저장

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (N, C, H, W) shape
        """
        self.x = x  # 입력 저장
        batch_size, in_channels, in_height, in_width = x.shape

        # 패딩 추가
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # 출력 크기 계산
        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        # 출력 텐서 초기화: (batch_size, output_channels, out_height, out_width)
        out = np.zeros((batch_size, self.output_channels, out_height, out_width))

        # 슬라이딩 윈도우 방식의 합성곱 연산 수행
        for yi in range(out_height):
            for xi in range(out_width):
                # 입력 데이터에서 슬라이딩 윈도우 패치 추출
                h_start = yi * self.stride
                h_end = h_start + self.kernel_size
                w_start = xi * self.stride
                w_end = w_start + self.kernel_size

                # 입력의 해당 영역에 대해 필터 적용 (벡터화 연산)
                patch = x[:, :, h_start:h_end, w_start:w_end]
                
                # 필터를 적용하여 출력 계산
                out[:, :, yi, xi] = np.tensordot(patch, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b.flatten()

        return out

    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        grad: (batch_size, output_channels, out_height, out_width) 형태의 출력에 대한 손실 기울기
        """
        batch_size, out_channels, out_height, out_width = grad.shape
        _, in_channels, in_height, in_width = self.x.shape

        # 패딩 추가된 입력 데이터에 대한 그라디언트 계산
        if self.padding > 0:
            x_padded = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = self.x

        # 필터에 대한 기울기와 입력에 대한 기울기 초기화
        dW = np.zeros_like(self.W)  # 필터에 대한 기울기
        db = np.zeros_like(self.b)  # 바이어스에 대한 기울기
        dx_padded = np.zeros_like(x_padded)  # 입력에 대한 기울기

        # 역전파 슬라이딩 윈도우 방식으로 기울기 계산
        for yi in range(out_height):
            for xi in range(out_width):
                h_start = yi * self.stride
                h_end = h_start + self.kernel_size
                w_start = xi * self.stride
                w_end = w_start + self.kernel_size

                # 현재 출력 기울기(grad)의 위치에 있는 값
                grad_value = grad[:, :, yi, xi][:, :, np.newaxis, np.newaxis, np.newaxis]  # (batch_size, output_channels, 1, 1, 1)

                print(f"grad_value.shape ")

                # 필터의 기울기 계산
                patch = x_padded[:, :, h_start:h_end, w_start:w_end]  # (batch_size, in_channels, kernel_size, kernel_size)
                dW += np.tensordot(grad_value, patch, axes=([0], [0]))  # 필터의 기울기 계산

                # 바이어스 기울기 계산
                db += np.sum(grad_value, axis=(0, 2, 3, 4))

                # 입력에 대한 기울기 계산
                dx_padded[:, :, h_start:h_end, w_start:w_end] += np.tensordot(grad_value, self.W, axes=([1], [0]))

        # 패딩이 있으면 패딩 부분을 제거
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded

        # 가중치 및 바이어스 업데이트
        self.W -= learning_rate * dW / batch_size
        self.b -= learning_rate * db / batch_size

        return dx


    def conv2d_fft(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        2D 합성곱을 FFT 기반으로 계산하는 함수
        image: 입력 이미지 (2D numpy 배열)
        kernel: 필터/커널 (2D numpy 배열)
        """
        # 이미지와 커널의 크기를 합성곱 결과에 맞게 조정 (Zero padding)
        image_size = image.shape
        kernel_size = kernel.shape
        
        # Zero-padding을 통해 결과의 크기를 맞추기
        padded_size = [image_size[0] + kernel_size[0] - 1, image_size[1] + kernel_size[1] - 1]
        
        # FFT 적용 (2D 푸리에 변환)
        fft_image = np.fft.fft2(image, padded_size)
        fft_kernel = np.fft.fft2(kernel, padded_size)
        
        # 주파수 도메인에서 곱셈 수행
        fft_conv_result = fft_image * fft_kernel
        
        # 역 FFT 적용 (결과를 다시 공간 도메인으로 변환)
        conv_result = np.fft.ifft2(fft_conv_result)
        
        # 결과의 실수 부분을 취함 (복소수 연산 결과로 실수/허수가 섞일 수 있음)
        conv_result = np.real(conv_result)
        
        # 최종 결과는 원본 이미지 크기와 커널 크기의 합에 해당하는 크기이므로 이를 잘라냄
        return conv_result

    def winograd_conv2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Winograd 알고리즘 기반으로 2D 합성곱을 수행하는 함수 (F(2x2, 3x3))
        image: 입력 이미지 (2D numpy 배열)
        kernel: 3x3 필터 (2D numpy 배열)
        """
        if kernel.shape != (3, 3):
            raise ValueError("Winograd F(2x2, 3x3)에서는 필터 크기가 3x3이어야 합니다.")

        # Output의 크기는 이미지보다 작아짐 (F(2x2, 3x3))
        out_height = image.shape[0] - 2
        out_width = image.shape[1] - 2
        
        # 출력 텐서 초기화
        output = np.zeros((out_height, out_width))

        # Winograd 알고리즘에서 사용되는 고정 행렬들
        G = np.array([[1, 0, 0],
                    [0.5, 0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0, 0, 1]])

        BT = np.array([[1, 0, -1],
                    [0, 1, 1],
                    [-1, 1, 0]])

        AT = np.array([[1, 1, 1],
                    [1, -1, 1]])

        # 필터 변환 (G @ kernel @ G.T)
        kernel_transformed = G @ kernel @ G.T

        # 각 2x2 블록마다 Winograd 변환을 수행
        for i in range(out_height):
            for j in range(out_width):
                # 이미지에서 4x4 블록을 추출
                block = image[i:i+4, j:j+4]
                
                # 입력 변환 (BT @ block @ B)
                block_transformed = BT @ block @ BT.T

                # 변환된 입력과 필터의 요소별 곱셈
                V = block_transformed * kernel_transformed

                # 출력 변환 (A.T @ V @ A)
                output[i, j] = np.sum(AT @ V @ AT.T)

        return output


class MaxPooling(Module):
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size  # 풀링 영역의 크기 (예: 2x2)
        self.stride = stride  # 스트라이드 크기
        self.x = None  # 입력 데이터 저장
        self.argmax = None  # 각 풀링 영역에서 최대값의 위치를 저장

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        MaxPooling의 순전파
        x: (batch_size, channels, height, width) 형태의 입력
        """
        self.x = x
        batch_size, channels, height, width = x.shape

        # 출력 크기 계산
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        # 출력 텐서 초기화
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.argmax = np.zeros_like(x, dtype=int)  # 최대값의 위치를 저장할 배열

        # MaxPooling 연산
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        # 풀링 영역에서 최대값 추출
                        pooling_region = x[b, c, h_start:h_end, w_start:w_end]
                        max_value = np.max(pooling_region)
                        output[b, c, h, w] = max_value

                        # 최대값 위치 저장 (역전파를 위해)
                        max_index = np.argmax(pooling_region)
                        max_pos = np.unravel_index(max_index, pooling_region.shape)
                        self.argmax[b, c, h_start + max_pos[0], w_start + max_pos[1]] = 1

        return output

    def backward(self, grad: np.ndarray, learning_rate:float) -> np.ndarray:
        """
        MaxPooling의 역전파
        grad: (batch_size, channels, out_height, out_width) 형태의 기울기
        """
        batch_size, channels, out_height, out_width = grad.shape

        # 입력 크기와 동일한 배열로 기울기 초기화
        dx = np.zeros_like(self.x)

        # 역전파 (최댓값 위치로만 기울기를 전파)
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        # 역전파: 최대값 위치로 기울기 전파
                        dx[b, c, h_start:h_end, w_start:w_end] += (
                            self.argmax[b, c, h_start:h_end, w_start:w_end] * grad[b, c, h, w]
                        )

        return dx

class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        return (self.x > 0) * grad