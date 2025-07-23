import torch
from model.embedding import Embedding
from model.RNN_Attention import AttentionRNN

# 동일한 설정
word_num = 100
embedding_dim = 32
hidden_size = 8
output_size = 1
seq_len = 7

# 모델 재정의 + 가중치 불러오기
embedding = Embedding(word_num, embedding_dim)
rnn = AttentionRNN(embedding_dim, hidden_size, output_size)

checkpoint = torch.load('model_weights.pth')
embedding.load_state_dict(checkpoint['embedding'])
rnn.load_state_dict(checkpoint['rnn_attention'])

# 예측
embedding.eval()
rnn.eval()

# 5개의 더미 테스트 데이터셋
test_tokens = torch.randint(0, word_num, (5, seq_len))
test_embed = embedding(test_tokens)

with torch.no_grad():
    preds = rnn(test_embed)

print("예측 결과:")
for i, pred in enumerate(preds):
    print(f"샘플 {i+1}: {pred.item():.4f}")
