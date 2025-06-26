import torch
import torch.nn as nn
from einops import rearrange

class Attention(nn.Module):
    def __init(self, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.head_dim = config['head_dim']
        self.emb_dim = config['emb_dim']
        self.drop_prob = config['dropout'] if 'dropout' in config else 0.0
        self.att_dim = self.n_heads * self.head_dim
        
        self.qkv_proj = nn.Linear(self.emb_dim, 3 * self.att_dim, bias=False) #ma trận D x (3 x 8 x head_dim), big W matrix
        self.output_proj = nn.Sequential(
            nn.Linear(self.att_dim, self.emb_dim),  
            nn.Dropout(self.drop_prob) #ma trận cuối cùng đưa vào đầu ra của attention về patches x emb_dim
        )
        self.attn_dropout = nn.Dropout(self.drop_prob)
        
    def forward(self, x):
        # x: [batch_size, n_patches, emb_dim]
        B, N = x.shape[:2]
        
        #chiếu để biến thành 3*att_dim để chia thành q, k, v
        #qkv - [batch_size, n_patches, 3 * att_dim (n_heads * head_dim)] (ma trận này đang là hợp của q, k, v mà chưa tách ra nên 3 * att_dim)
        #q, k, v - [batch_size, n_patches, att_dim] (q, k, v to hợp của các head khác nhau) (đã tách q, k, v ra nên chia 3)
        q, k, v = self.qkv_proj(x).split(self.att_dim, dim=-1) #chia q, k, v
         #[batch_size, n_patches, att_dim]
         #-> [batch_size, n_patches, n_heads * head_dim]
         #-> [batch_size, n_heads, n_patches, head_dim]
         #-> [B, H, N, head_dim]
        q = rearrange(q, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        k = rearrange(k, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)    
        v = rearrange(v, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        
        #Tính trọng số Attention
        # B x H x N x Head Dimension @ B x H x Head Dimension x N
        # -> B x H x N x N (NxN là n_patches x n_patches) (có 8 head chồng lên nhau, mỗi head có một ma trận QK^T riêng cỡ NxN)
        att = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**(-0.5)) #tính tích ma trận QK^T cho từng head song song
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Tính trọng số Value
        #########################################################
        #  B x H x N x N @ B x H x N x Head Dimension
        # -> B x H x N x Head Dimension
        out = torch.matmul(att, v)
        #########################################################
        
        # Chiếu trở về số chiều của đầu vào 
        #########################################################
        # B x N x (Heads * Head Dimension) -> B x N x (Attention Dimension)
        out = rearrange(out, 'b n_h n h_dim -> b n (n_h h_dim)')
        #  B x N x Dimension
        out = self.output_proj(out)
        ##########################################################
        
        return out
         
        
        