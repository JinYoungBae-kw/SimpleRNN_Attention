import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size

        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

        # 초기화
        nn.init.xavier_uniform_(self.input_to_hidden.weight)
        nn.init.zeros_(self.input_to_hidden.bias)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)
        nn.init.zeros_(self.hidden_to_output.bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        all_hidden_states = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, h), dim=1)
            h = self.tanh(self.input_to_hidden(combined))
            # 모든 시점의 h 리스트
            all_hidden_states.append(h.unsqueeze(1))

        # 모든 시점의 h 텐서
        hidden_seq = torch.cat(all_hidden_states, dim=1)
        # 현 시점 (혹은 최종) h 값
        final_hidden = hidden_seq[:, -1, :]

        # (hi, ht) 내적
        scores = torch.bmm(hidden_seq, final_hidden.unsqueeze(2)).squeeze(2)
        # attention_score 구하기
        attn_weights = F.softmax(scores, dim=1)

        # attention_score * hi 를 모두 더해서 c 구하기
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_seq).squeeze(1)

        y = self.hidden_to_output(context)
        return y
