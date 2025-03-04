import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd,n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd,n_embd)
        
        # regularization
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout=nn.Dropout(0.1)
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y = self.dropout(y)
        #print(f"this is for self : {y.shape}")
        return y
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd,ff_hid_dim):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, ff_hid_dim)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(ff_hid_dim, n_embd)
        
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
       
        
        return x
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head,ff_fid_dim):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = CausalSelfAttention(n_embd,n_head)
        self.ffwd = FeedFoward(n_embd,ff_fid_dim)
        
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        
        x = x + self.ffwd(self.ln2(x))
        return x


class Vec2Data(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet=nn.Conv2d(64,3,kernel_size=(1,1)).to("cuda")
        self.num_tokens = int(128**2/16**2)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens+1, 64))
        self.encoder_layers = nn.ModuleList([
            Block(64,8,32)
            for _ in range(6)
        ]).to("cuda")
    

    def forward(self,inputs):
        b,n,e=inputs.shape
        x = inputs + self.pos_embedding[:, :n]  
        for layer in self.encoder_layers:  # Katmanları sırayla uygula
            x = layer(x)
        x=x.permute(0,2,1).reshape(b,64,8,8)
        x=self.convnet(x)
        x=nn.functional.interpolate(x,(128,128),mode="bilinear")
        x=torch.sigmoid(x)
        return x

