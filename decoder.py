import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class Decoder(nn.Module):
    def __init__(self,
                 depth,
                 num_tokens,
                 embed_dim,
                 decoder_dim,
                 kernel_size,
                 padding,
                 groups,
                 ):
        super().__init__()
        #back calculate the height and width of the image
        self.h, self.w = int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens))
        
        #create a list of layers
        self.convs = nn.ModuleList()
        
        #add the first layer, converting to the decoder dimension (b x embed_dim x h x w -> b x decoder_dim x h x w)
        self.convs.append(nn.Conv2d(embed_dim, decoder_dim, kernel_size=kernel_size, padding=padding, groups=groups))
        self.convs.append(nn.LayerNorm((decoder_dim, self.h, self.w)))
        self.convs.append(nn.GELU())
        
        #add the remaining layers
        for i in range(depth - 1):
            self.convs.append(nn.Conv2d(decoder_dim, decoder_dim, kernel_size=kernel_size, padding=padding, groups=groups))
            self.convs.append(nn.LayerNorm((decoder_dim, self.h, self.w)))
            self.convs.append(nn.GELU())
        
        #project back to the embedding dimension
        self.proj = nn.Linear(decoder_dim, embed_dim)
        
    def forward(self, x):
        
        #reshape the input to the correct dimensions, as implemented in the fairseq code
        b,m,c=x.shape
        x=x.permute(0,2,1).reshape(b,c,self.h,self.w)
       
        
        #use a residual connection
        residual = x
        
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i > 0:
                x = x + residual
            residual = x
        
        #project back to the embedding dimension
        x=x.flatten(2).permute(0,2,1)
        
        x = self.proj(x)
        return x
