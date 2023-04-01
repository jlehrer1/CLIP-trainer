import torch.nn as nn
import torch.nn.functional as F
import torch 

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, feed_forward_dim, dropout):
        """
        The text encoder for the CLIP model

        :param embedding_dim: the dimension of the embedding
        :param num_heads: the number of heads in the multihead attention layer
        :param feed_forward_dim: the dimension of the feed forward layer
        :param dropout: the dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dropout = dropout

        self.attention = nn.MultiheadAttention(self.embedding_dim, self.num_heads, self.dropout)
        self.linears = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(4)])
        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embedding_dim, self.feed_forward_dim),
            nn.ReLU(),
            nn.Linear(self.feed_forward_dim, self.embedding_dim)
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder layer
        """
        x = self.norm1(x)
        x = self.attention(self.linears[0](x), self.linears[1](x), self.linears[2](x))[0]
        x = self.dropout(x)
        x = x + self.linears[3](x)
        x = self.norm2(x)
        x = x + self.feed_forward(x)

        return x
    

class TextEncoder(nn.Module):
    def __init__(self, num_heads, num_layers, max_len, vocab_size, embedding_dim, feed_forward_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.feed_forward_dim = feed_forward_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pos_embedding = nn.Embedding(self.max_len, self.embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(self.embedding_dim, self.num_heads, self.feed_forward_dim, self.dropout) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        Forward pass for the text encoder
        """
        x = self.embedding(x) * (self.embedding_dim ** 0.5)
        x += self.pos_embedding(torch.arange(x.shape[1]).to(x.device))
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)

        # We need the output shape to be (batch_size, embedding_dim)
        # to be able to calculate cosine similarity with the image encoder
        # However, the output shape of the encoder is (batch_size, max_len, embedding_dim)
        # so we need to remove the max_len dimension somehow
        # we do this by taking the mean of the max_len dimension
        x = torch.mean(x, dim=1)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)