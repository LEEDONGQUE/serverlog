import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 1. PPL 계산 함수 (동규님 기존 코드 유지)
def compute_ppl(model, x):
    model.eval()
    input_seq = x[:, :-1]
    target_seq = x[:, 1:]

    with torch.no_grad():
        output = model(input_seq)
        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            target_seq.reshape(-1)
        )
    return math.exp(loss.item())

# 2. LSTM 모델 구조 (동규님 기존 코드 유지)
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# 3. 🚨 [태훈님 요청] 이중 for문 기반 정밀 이상치 탐지 함수
def analyze_anomalies_detailed(model, data, threshold=20.0):
    """
    바깥쪽 for문: 전체 데이터 샘플 탐색
    안쪽 for문: 슬라이딩 윈도우 내부의 시점별 이상치(Anomaly) 결정
    """
    model.eval()
    device = next(model.parameters()).device
    results = []

    print(f"[*] 정밀 분석 시작 (총 {len(data)}개 샘플)...")

    # [바깥쪽 FOR문]: 데이터 개수만큼 반복
    for i in range(len(data)):
        sample = torch.LongTensor(data[i]).unsqueeze(0).to(device) # (1, Window_Size)
        
        # 샘플 전체에 대한 기본 PPL 계산
        overall_ppl = compute_ppl(model, sample)
        
        # 이상치로 의심될 경우 내부를 더 자세히 들여다봄
        if overall_ppl > threshold:
            print(f"\n[!] 이상 징후 발견 (Sample {i}) | 전체 PPL: {overall_ppl:.2f}")
            
            # [안쪽 FOR문]: 슬라이딩 윈도우 크기만큼 내부 탐색
            # 어느 지점에서 예측 에러가 커지는지 확인
            window_size = sample.size(1)
            for t in range(1, window_size):
                context = sample[:, :t]    # 지금까지 본 로그들
                target = sample[:, t]       # 다음에 올 로그 (정답)
                
                with torch.no_grad():
                    output = model(context)
                    # 마지막 시점의 예측값만 추출
                    prediction = output[:, -1, :]
                    # 개별 로그에 대한 Loss 계산
                    step_loss = F.cross_entropy(prediction, target).item()
                    step_ppl = math.exp(step_loss)
                    
                    # 특정 로그 단계에서 갑자기 당황(PPL 급증)했다면 출력
                    if step_ppl > threshold:
                        print(f"   -> [시점 {t}] 위험 로그 탐지! 로그ID: {target.item()} | 지점 PPL: {step_ppl:.2f}")
            
            results.append((i, overall_ppl))

    return results

# 테스트 및 실행 블록
if __name__ == "__main__":
    # 설정값
    vocab_size = 200 # id_map.json 크기에 맞춰 조정 가능
    model = LSTMLM(vocab_size)
    
    # 만약 학습된 가중치가 있다면 로드 (파일이 있을 때만 활성화)
    # model.load_state_dict(torch.load('lstm_final.pth'))
    
    # 가상의 데이터 (10개 샘플, 윈도우 크기 11)
    dummy_data = np.random.randint(0, vocab_size, (10, 11))
    
    # 정밀 분석 실행
    analyze_anomalies_detailed(model, dummy_data, threshold=15.0)
