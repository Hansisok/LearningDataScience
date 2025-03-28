import torch
import torch.nn as nn
import torch.nn.functional as F

# 小词表（模拟）
vocab = ["<pad>", "<bos>", "<eos>", "The", "cat", "sat", "on", "mat", "."]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

vocab_size = len(vocab)
embedding_dim = 32
max_len = 10

# 简单 GPT 模型（Transformer Decoder）
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(max_len, embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=2)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(0)
        positions = torch.arange(seq_len).unsqueeze(1)
        x_embed = self.embed(x) + self.pos_embed(positions)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        output = self.decoder(tgt=x_embed, memory=torch.zeros_like(x_embed), tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

# 初始化模型
model = MiniGPT()
model.eval()

# 自动续写函数
def generate_text(start_words, max_gen=5):
    tokens = [word2idx["<bos>"]] + [word2idx[w] for w in start_words]
    input_seq = torch.tensor(tokens)

    for _ in range(max_gen):
        with torch.no_grad():
            logits = model(input_seq)
            next_token = torch.argmax(logits[-1], dim=-1).item()
        if next_token == word2idx["<eos>"]:
            break
        input_seq = torch.cat([input_seq, torch.tensor([next_token])])

    return [idx2word[idx.item()] for idx in input_seq[1:]]

# 示例：从 "The cat" 开始自动生成
generated = generate_text(["The", "cat"])
print("生成结果:", " ".join(generated))
