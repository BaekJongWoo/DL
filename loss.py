import numpy as np

class Loss:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError()

class CrossEntropyLoss:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        y: (batch_size,) 실제 레이블 (정수 인덱스 형식, 클래스 인덱스)
        y_pred: (batch_size, num_classes) 모델의 출력 값 (로짓)
        
        Returns:
        float: 평균 cross-entropy 손실 값
        """
        # 1. Softmax를 적용하여 예측 값을 확률로 변환
        exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))  # 안정성 확보 (overflow 방지)
        softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # 2. 실제 레이블에 해당하는 예측 확률을 선택
        batch_size = y_pred.shape[0]
        correct_class_probs = softmax_pred[np.arange(batch_size), y]
        
        # 3. Cross-entropy 손실 계산 (-log(p))
        loss = -np.log(correct_class_probs)
        
        # 4. 배치의 평균 손실 반환
        return np.mean(loss)