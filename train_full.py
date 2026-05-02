import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
from lstm_model import LSTMLM, compute_ppl # 동규님의 모델 클래스 임포트

# 1. 데이터 로드 및 전처리
print("[*] 데이터를 불러오는 중...")
X_data = np.load('X_train.npy') # (19990, 11) 형태

with open('id_map.json', 'r') as f:
    id_map = json.load(f)
    vocab_size = len(id_map) + 1 # ID가 1부터 시작할 경우 대비 +1

# 2. 데이터 분할 (16,000개 학습 / 2,000개 검증)
# X_train.npy의 마지막 컬럼이 정답(Target)입니다.
X_train = torch.LongTensor(X_data[:16000, :-1])
y_train = torch.LongTensor(X_data[:16000, -1])

X_val = torch.LongTensor(X_data[16000:18000, :]) # PPL 계산용 (입력+타겟 포함 전체)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 3. 장치 설정 (RTX 2070 활용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] 사용 중인 장치: {device}")

# 4. 모델 초기화
model = LSTMLM(vocab_size=vocab_size, embed_size=64, hidden_size=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 루프
epochs = 10
print(f"[*] 총 {epochs} 에폭 학습을 시작합니다.")

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_X)
        
        # LSTM의 마지막 시점 출력만 사용하여 다음 로그 예측
        loss = criterion(output[:, -1, :], batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 에폭 종료 후 검증 PPL 계산
    avg_loss = total_loss / len(train_loader)
    val_ppl = compute_ppl(model, X_val.to(device))
    epoch_time = time.time() - start_time
    
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val PPL: {val_ppl:.4f} | 시간: {epoch_time:.2f}s")

# 6. 최종 모델 저장
torch.save(model.state_dict(), 'lstm_final.pth')
print("-" * 30)
print("[+] 학습 완료! 'lstm_final.pth' 저장이 완료되었습니다.")
