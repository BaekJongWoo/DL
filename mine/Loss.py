import numpy as np

class Loss:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> tuple:
        raise NotImplementedError()

class CrossEntropyLoss:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> tuple:
        softmax_pred = self.softmax(y_pred)
        correct_class_probs = np.sum(y * softmax_pred, axis=1)
        
        loss = -np.log(correct_class_probs)
        loss = np.mean(loss)
        
        grad = softmax_pred - y
        grad = grad / y.shape[0]

        return loss, grad
    
    def softmax(self, y_pred: np.ndarray) -> np.ndarray:
        exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))  # overflow 방지
        softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        return softmax_pred