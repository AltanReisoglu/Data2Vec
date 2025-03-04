import torch

import torch.nn as nn

from x_transformers import Encoder as XEncoder  # Çakışmayı önlemek için isim değiştirdik

class Encoder(nn.Module):
    def __init__(self, embedding=64, depth=4, num_heads=8):
        super(Encoder, self).__init__()

        # x_transformers Encoder kullanımı
        self.encoder = XEncoder(
            dim=embedding,  # Dönüştürücü boyutu
            depth=depth,    # Transformer katman sayısı
            heads=num_heads # Multi-head attention başlık sayısı
        )

    def forward(self, inputs , is_teacher=False):
        
        if is_teacher:
            _,output=self.encoder(inputs,return_hiddens=True)
            output = torch.stack([h.clone() for h in output.hiddens])
            
            return output
        else:
            return self.encoder(inputs)



        
            