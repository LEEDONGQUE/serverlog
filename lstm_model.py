import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def compute_ppl(model, x):
    model.eval()

    # 입력 / 타겟 분리 (shift)
    input_seq = x[:, :-1]
    target_seq = x[:, 1:]

    with torch.no_grad():
        output = model(input_seq)

        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            target_seq.reshape(-1)
        )

    return math.exp(loss.item())
# LSTM Language Model
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


# 테스트 실행
if __name__ == "__main__":
    vocab_size = 10
    model = LSTMLM(vocab_size)

    # 더미 데이터 (batch=1, 길이=5)
    x = torch.tensor([[1, 2, 3, 4, 5]])

    output = model(x)
    print("output shape:", output.shape)
if __name__ == "__main__":
    vocab_size = 10
    model = LSTMLM(vocab_size)

    x = torch.tensor([[1, 2, 3, 4, 5]])

    ppl = compute_ppl(model, x)
    print("PPL:", ppl)
