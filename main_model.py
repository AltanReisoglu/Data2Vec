from datasets import load_dataset
import torch
from torch import nn
from patch_embdd import PatchEmbed
from encoder import Encoder
from decoder import Decoder
from torch.nn import functional as F


class data2vec_base(nn.Module):
    def __init__(
            self,
            img_size=128, #img_size is the height and width of the image
            patch_size=16, #patch_size is the height and width of the patches
            in_chans=3, #in_chans is the number of channels in the image
            embed_dim=64, #embed_dim is the embedding dimension of the patches/tokens
            masking_ratio=0.6, #masking_ratio is the ratio of tokens to mask
            heads=8, #heads is the number of attention heads in the encoder
            depth=8, #depth is the number of layers in the encoder
            decoder_depth=3, #decoder_depth is the number of layers in the decoder
            decoder_dim=64, #decoder_dim is the dimension of the decoder
            decoder_kernel_size=3, #decoder_kernel_size is the kernel size of the decoder
            decoder_padding=1, #decoder_padding is the padding of the decoder
            decoder_groups=1, #decoder_groups is the number of groups in the decoder
            post_emb_norm=True, #post_emb_norm is whether to apply layer normalization after the embedding
            dropout=0., #dropout is the dropout rate
            is_teacher = False, #is_teacher is whether the model is being used as a teacher
            k=4, #k is the number of transformer blocks to use as the teacher's 'context' for target creation
    ):
        super().__init__()
        if is_teacher:
            assert(k > 0 and masking_ratio == 0)
        
        #define the parameters
        self.masking_ratio = masking_ratio
        self.is_teacher = is_teacher
        self.k = k
        self.num_tokens = int(img_size**2/patch_size**2)
        
        #define the patch embedding and positional embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens+1, embed_dim))
        
        #define the cls and mask tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, 0.02)
        nn.init.trunc_normal_(self.mask_token, 0.02)

        #define the encoder and decoder, as well as the layer normalization and dropout
        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder().to(device="cuda")    
        self.decoder = Decoder(depth=decoder_depth, num_tokens=self.num_tokens, embed_dim=embed_dim, decoder_dim=decoder_dim, kernel_size=decoder_kernel_size, padding=decoder_padding, groups=decoder_groups).to("cuda")
    
    #generate the targets for the teacher model
    @torch.no_grad()
    def generate_targets(self, x:torch.Tensor, encoder: nn.Module, k:torch.Tensor=4):
        encoder = encoder.eval() #not sure if this is necessary
        
        intermediates = encoder(x, is_teacher=True) #get intermediates from the encoder
        
        
        intermediates = intermediates[-k:] #top k hidden states
        b, n, h, w = intermediates.shape
        intermediates = intermediates.flatten(start_dim=0,end_dim=1)   
        #normalize the hidden states
        
        
        intermediates = F.instance_norm(intermediates)
        
        intermediates = intermediates.reshape(b,n,h,w)
        intermediates = intermediates.mean(0)
        
        return intermediates[:, 1:] #return non cls token
    
    def forward(self, x):
        
        #get the patch embeddings
        x = self.patch_embed(x)
        b, n, e = x.shape

        #add positional embedding
        x = x + self.pos_embedding[:, :n].to("cuda")
        
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked - adapted from lucidrains' implementation of MAE
        num_masked = int(self.masking_ratio * n)
        rand_indices = torch.rand(b, n).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        # get the unmasked tokens to be encoded

        batch_range = torch.arange(b)[:, None]
        x = x[batch_range, unmasked_indices]
        #add cls and embedding tokens
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        #post embedding norm
        x = self.post_emb_norm(x)
        
        #get target representations if teacher
        if self.is_teacher:
            return self.generate_targets(x, self.encoder,self.k)
        
        #x shape: (b, n + 1, e)
        
        #encode the tokens
        x = self.dropout(x)
        x = self.encoder(x,is_teacher=False)
        x = self.norm(x)

        #reconstruct the tokenss
        reconstruced = torch.zeros(b, n, e, dtype=x.dtype).to("cuda")
        cls_embedding = x[:, 0]
        reconstruced[batch_range, unmasked_indices] = x[:, 1:]
        reconstruced[batch_range, masked_indices] = self.mask_token 
        reconstruced.type_as(x)
        #reconstructed shape: (b, n, e)

        #decode the tokens
        decoded = self.decoder(reconstruced)
        
        return decoded
