from dataset.dataloader import Dataloader
from mine.model import Sequantial, Model
from mine.module import Conv2D, Flatten, Linear, MaxPooling, ReLU
from mine.loss import CrossEntropyLoss, softmax

from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def MyNN():
    h1_size = 256
    h2_size = 128
    model = Sequantial([
        Flatten(),
        Linear(28*28, h1_size),
        ReLU(),
        Linear(h1_size, h2_size),
        ReLU(),
        Linear(h2_size, 10)
    ])
    return model

def MyCNN():
    model = Sequantial([                # n 1 28 28
        Conv2D(1, 3, kernel_size=3),    # n 3 26 26
        ReLU(),
        MaxPooling(),                   # n 3 13 13
        Conv2D(3, 5, kernel_size=6),    # n 5 8 8
        ReLU(),
        MaxPooling(),                   # n 5 4 4
        Flatten(),                      # n 80
        Linear(80, 10),
    ])
    return model

def drawLosses(train_loss_values, test_loss_values):
    loss_values = np.array(train_loss_values)
    n = 100
    term = len(loss_values) // n
    split_values = np.split(loss_values, n)
    mean_values = np.mean(split_values, axis=1)
    x_values = np.arange(0, len(loss_values), term)

    # Plot the Train Loss with Mean ± Standard Deviation
    plt.figure()
    plt.plot(x_values, mean_values, label="Train Loss", alpha=0.6, color='b')

    loss_values = np.array(test_loss_values)
    n = 100
    term = len(loss_values) // n
    split_values = np.split(loss_values, n)
    mean_values = np.mean(split_values, axis=1)
    x_values = np.arange(0, len(loss_values), term)

    # Plot the Train Loss with Mean ± Standard Deviation
    plt.plot(x_values, mean_values, label="Test Loss", alpha=0.6, color='r')

    # plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss over Iteration")
    plt.legend()
    plt.savefig(f"loss_graph.png")
    plt.close()

def train(model: Model, train_dataloader: Dataloader, test_dataloader: Dataloader, epoch_num: int, learning_rate: float):
    
    loss_fn = CrossEntropyLoss()
    
    train_loss_values = []
    test_loss_values = []
    
    for epoch in range(epoch_num):
        total_train_loss = 0
        total_test_loss = 0
        last_update_time = time.time()
        
        # Training loop
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for data_idx in range(len(train_dataloader)):
                data = train_dataloader[data_idx]
                x, y = data
                y_pred = model.forward(x)
                loss, grad = loss_fn(y, y_pred)
                model.backward(grad, learning_rate)

                train_loss_values.append(loss)
                total_train_loss += loss


                test_idx = data_idx % len(test_dataloader)
                test_data = test_dataloader[test_idx]
                tx, ty = test_data
                ty_pred = model.forward(tx)
                print(ty_pred)
                tloss, _ = loss_fn(ty, ty_pred, no_grad=True)

                test_loss_values.append(tloss)
                total_test_loss += tloss

                if time.time() - last_update_time >= 1:
                    pbar.set_postfix(loss=f"{loss:7.4f}")
                    last_update_time = time.time()
                pbar.update()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_test_loss = total_test_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Train Loss: {avg_train_loss:.4f}, Averge Test Loss: {avg_test_loss:.4f}")

        drawLosses(train_loss_values, test_loss_values)
        
def PrintCM(model: Model, test_dataloader: Dataloader):
    y_true = []
    y_pred = []
    
    print("\nGenerating Confusion Matrix...")
    for test_data in tqdm(test_dataloader):
        tx, ty = test_data
       
        ty_pred = model.forward(tx)

        y_true.extend(np.argmax(ty, axis=1))
        y_pred.extend(np.argmax(ty_pred, axis=1))

    # Confusion Matrix 생성
    cm = confusion_matrix(y_true, y_pred)
    
    # Confusion Matrix 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"confusion_matrix.png")
    plt.close()

    print("Complete.")


def PrintTop3(model: Model, test_dataloader: Dataloader):
    # 각 숫자 (0-9)에 대해 상위 3개의 확률과 이미지를 저장할 배열
    top3_prob = np.zeros((10, 3))  # 각 클래스의 상위 3개 확률을 저장할 배열
    top3_image = np.zeros((10, 3), dtype=object)  # 각 클래스의 상위 3개 이미지를 저장할 배열

    print("\nGenerating Top 3 Images")
    for test_data in tqdm(test_dataloader, desc="Finding Top 3 Images"):
        tx, ty = test_data  # tx: 이미지, ty: 실제 라벨

        # 모델로 예측
        ty_pred = model.forward(tx)

        ty_pred_probs = np.max(softmax(ty_pred), axis=1)
        ty_pred_index = np.argmax(ty_pred, axis=1)

        # 각 이미지에 대한 예측 확률
        for idx in range(len(ty_pred_index)):
            pred_prob = ty_pred_probs[idx]
            pred_index = ty_pred_index[idx]
            image = tx[idx]  # 해당 이미지

            # 만약 현재 확률이 상위 3개에 포함된다면 업데이트
            for i in range(3):
                if pred_prob > top3_prob[pred_index, i]:
                    # i번째 확률보다 높다면, 해당 확률과 이미지를 그 자리에 넣고, 나머지 하나씩 밀기
                    top3_prob[pred_index, i+1:] = top3_prob[pred_index, i:-1]
                    top3_image[pred_index, i+1:] = top3_image[pred_index, i:-1]

                    top3_prob[pred_index, i] = pred_prob
                    top3_image[pred_index, i] = image
                    break

    fig, axes = plt.subplots(10, 3, figsize=(12, 30))  # 10행 3열의 서브플롯
    fig.suptitle("Top 3 Images for Each Digit", fontsize=16)

    # 각 숫자에 대해 상위 3개의 이미지를 출력
    for num in range(10):
        for i in range(3):
            if top3_image[num, i] is not None:
                axes[num, i].imshow(top3_image[num, i].squeeze(), cmap='gray')
                axes[num, i].set_title(f"Prob: {top3_prob[num, i]:.4f}")
                axes[num, i].axis('off')
                
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"top_3_images.png")
    plt.close()

    print("Complete")

if __name__ == "__main__":
    batch_size = 50
    epoch_num = 1
    learning_rate = 1e-2

    train_dataloader = Dataloader('dataset', is_train=True, batch_size=batch_size) # total 60000
    test_dataloader = Dataloader('dataset', is_train=False, batch_size=batch_size) # total 10000

    model = MyNN()

    train(model, train_dataloader, test_dataloader, epoch_num, learning_rate)

    PrintCM(model, test_dataloader)

    PrintTop3(model, test_dataloader)