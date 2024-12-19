import torch
import torch.nn as nn
import torch.nn.functional as F

# Define custom multi-head attention layer
class MultiAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super(MultiAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        # Define weights for query, key, and value projections
        self.WQ = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.WK = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.WV = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Ensure x is projected into the queries, keys, and values
        queries = self.WQ(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.WK(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.WV(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)

        # Concatenate the heads and reshape for output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return attention_output

# Define custom layer normalization
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Define residual connection with dropout and normalization
class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return x + self.dropout(self.norm(sublayer_output))

# Main model with custom attention and feedforward layers
# Main model with custom attention and feedforward layers
class TransformerModel(nn.Module):
    def __init__(self, num_genes, num_classes, num_heads=12, dim_feedforward=1920, dropout=0.3, embedding_dim=480):
        super(TransformerModel, self).__init__()
        
        # Ensure embedding_dim is divisible by num_heads
        assert embedding_dim % num_heads == 0, (
            f"Embedding dimension ({embedding_dim}) must be divisible by the number of heads ({num_heads})."
        )
        
        # Define the embedding layer and CLS token
        self.embedding = nn.Linear(num_genes, embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))  # Adjusted shape to match batch and seq length
        
        # Custom multi-head attention layers and residual connections
        self.multiattention1 = MultiAttention(num_heads, embedding_dim)
        self.residual_connection1 = ResidualConnection(embedding_dim, dropout)
        
        self.multiattention2 = MultiAttention(num_heads, embedding_dim)
        self.residual_connection2 = ResidualConnection(embedding_dim, dropout)
        
        self.multiattention3 = MultiAttention(num_heads, embedding_dim)
        self.residual_connection3 = ResidualConnection(embedding_dim, dropout)
        
        # Feedforward layers with dropout
        self.ffn1 = nn.Linear(embedding_dim, dim_feedforward)
        self.ffn2 = nn.Linear(dim_feedforward, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_feedforward = ResidualConnection(embedding_dim, dropout)
        
        # Output classification layer
        self.fc_out = nn.Linear(embedding_dim, num_classes)

    def feedforward(self, x):
        out = F.relu(self.ffn1(x))
        out = self.ffn2(self.dropout(out))
        return out

    def forward(self, x):
        # Generate embeddings for input data
        x = self.embedding(x).unsqueeze(1)  # Add sequence length dimension
        
        # Expand CLS token to match the batch size and concatenate
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (batch_size, 1, embedding_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate along sequence length dimension
        
        # Custom attention layers with residual connections
        attn_out1 = self.multiattention1(x)
        x = self.residual_connection1(x, attn_out1)
        
        attn_out2 = self.multiattention2(x)
        x = self.residual_connection2(x, attn_out2)
        
        attn_out3 = self.multiattention3(x)
        x = self.residual_connection3(x, attn_out3)
        
        # Feedforward layers with residual connection
        ff_out = self.feedforward(x)
        x = self.residual_feedforward(x, ff_out)
        
        # Output classification layer, using CLS token representation
        x = self.fc_out(x[:, 0])  # Only use the CLS token
        return x