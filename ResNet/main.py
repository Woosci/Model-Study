import torch
import torch.nn as nn
import torch.optim as optim
from model import resnet18  # 모델 불러오기
from cifar10 import get_cifar10_loaders  # 데이터 로더 불러오기
from train import train  # 학습 함수
from test import evaluate  # 평가 함수

# 하이퍼파라미터 설정
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Using device: {DEVICE}")

    # 데이터 로딩
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE, 
        resize=IMAGE_SIZE
    )

    # 모델 초기화
    model = resnet18(num_classes=NUM_CLASSES).to(DEVICE)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 모델 학습
    train(model, train_loader, criterion, optimizer, DEVICE, NUM_EPOCHS)

    # 모델 평가
    evaluate(model, test_loader, criterion, DEVICE)

if __name__ == "__main__":
    main()

