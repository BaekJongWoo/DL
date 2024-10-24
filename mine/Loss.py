import numpy as np

class Loss:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray, no_grad = False) -> tuple:
        raise NotImplementedError()

def softmax(y_pred: np.ndarray) -> np.ndarray:
    exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    return softmax_pred

class CrossEntropyLoss:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray, no_grad = False) -> tuple:
        softmax_pred = softmax(y_pred)
        correct_class_probs = np.sum(y * softmax_pred, axis=1)
        
        loss = -np.log(correct_class_probs)
        loss = np.mean(loss)
        
        grad = 0
        if not no_grad:
            grad = softmax_pred - y
            grad = grad / y.shape[0]

        return loss, grad
    

if __name__ == "__main__":
    arr = np.array([[100,500,900]])
    print(softmax(arr))
    arr = np.array([[0.1,0.5,0.9]])
    print(softmax(arr))