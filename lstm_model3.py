import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 1. PPL 계산 함수
def compute_ppl(model, x):
    model.eval()
    input_seq = x[:, :-1]
    target_seq = x[:, 1:]
    with torch.no_grad():
        output = model(input_seq)
        loss = F.cross_entropy(output.reshape(-1, output.size(-1)), target_seq.reshape(-1))
    return math.exp(loss.item())

# 2. LSTM 모델 구조
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

# 3. 정밀 분석 함수 (w값 추가 출력)
def analyze_anomalies_detailed(model, data, w_size, threshold=20.0):
    model.eval()
    device = next(model.parameters()).device
    results = []

    print(f"\n" + "="*60)
    print(f"[*] 실험 진행 중: 윈도우 크기(W) = {w_size} | 총 {len(data)}개 샘플")
    print("="*60)

    for i in range(len(data)):
        sample = torch.LongTensor(data[i]).unsqueeze(0).to(device)
        overall_ppl = compute_ppl(model, sample)

        if overall_ppl > threshold:
            print(f"\n[!] 이상 발견 (W={w_size}, Sample {i}) | 전체 PPL: {overall_ppl:.2f}")
            
            # 안쪽 for문: 시점별 탐색
            for t in range(1, sample.size(1)):
                context = sample[:, :t]
                target = sample[:, t]
                with torch.no_grad():
                    output = model(context)
                    prediction = output[:, -1, :]
                    step_loss = F.cross_entropy(prediction, target).item()
                    step_ppl = math.exp(step_loss)

                    if step_ppl > threshold:
                        print(f"   -> [T={t}] 위험! 로그ID: {target.item()} | PPL: {step_ppl:.2f}")

            results.append({"sample_idx": i, "max_ppl": overall_ppl})
    
    return results

# 4. 메인 실험 블록
if __name__ == "__main__":
    # 🚨 [태훈님 설정] 실험하고 싶은 윈도우 크기 4개를 넣으세요
    target_window_sizes = [5, 10, 15, 20]
    threshold_value = 15.0 # 이상치 기준
    vocab_size = 200 # id_map.json 크기

    # 모델 준비 및 가중치 로드
    model = LSTMLM(vocab_size)
    try:
        model.load_state_dict(torch.load('lstm_final.pth'))
        print("[+] 학습된 모델 가중치(lstm_final.pth)를 로드했습니다.")
    except:
        print("[!] 모델 파일을 찾을 수 없어 초기화 상태로 진행합니다.")

    # 실제 데이터 로드 (X_train.npy 파일이 있는 경우)
    try:
        raw_data = np.load('X_train.npy')
    except:
        # 파일이 없을 때만 가상 데이터 생성
        raw_data = np.random.randint(0, vocab_size, (10, 30))

    final_summary = []

    # 🚨 핵심: 윈도우 크기에 따른 루프 실행
    for w in target_window_sizes:
        # 데이터에서 앞에서부터 w 크기만큼만 슬라이싱하여 실험
        current_data = raw_data[:10, :w] 
        
        res = analyze_anomalies_detailed(model, current_data, w_size=w, threshold=threshold_value)
        
        if res:
            # 해당 윈도우 실험 중 가장 높게 튄 PPL 기록
            max_p = max([x['max_ppl'] for x in res])
            final_summary.append((w, max_p))
        else:
            final_summary.append((w, "None"))

    # 📊 최종 결과 요약 출력 (엑셀 복사용)
    print("\n" + "#"*60)
    print("### 최종 실험 결과 요약 ###")
    print(f"{'Window Size':<15} | {'Max PPL Found':<15}")
    print("-" * 35)
    for w, p in final_summary:
        if isinstance(p, float):
            print(f"{w:<15} | {p:<15.2f}")
        else:
            print(f"{w:<15} | {p:<15}")
    print("#"*60)
