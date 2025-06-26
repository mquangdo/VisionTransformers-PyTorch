import torch
import torch.nn as nn
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        #Ảnh có shape cxhxw: 3 x 224 x 224
        #patch h, w : 16, 16
        image_height = config['image_height']
        image_width = config['image_width']
        im_channels = config['image_channels']
        emb_dim = config['emb_dim']
        patch_emb_drop = config['patch_embd_drop']
        
        self.patch_height = config['patch_height']
        self.patch_width = config['patch_width']
        
        #tính toán số patch để khởi tạo tham số positional embedding
        num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)
        
        #đây là số chiều đầu vào của patch_embed_layer
        #sau khi patch ảnh 224 x 224 x 3 thì ảnh thành num_patches x patch_height x patch_width x im_channels = 3
        #-> 196 x 16 x 16 x 3
        patch_dim = im_channels * self.patch_height * self.patch_width #duỗi patch thành vector
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, emb_dim),  # ma trận W: patch_dim x emb_dim
        )
        
        #Thông tin positional cần được thêm vào cls do đó là 1 + num_patches 
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, emb_dim)) #cần thêm batch_size vào đầu nên đặt là 1
        self.cls_token = nn.Parameter(torch.randn(emb_dim))
        self.patch_embed_dropout = nn.Dropout(patch_emb_drop)# cls token
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # This is doing the B, 3, 224, 224 -> (B, num_patches, patch_dim) transformation
        # B, 3, 224, 224 -> B, 3, 14*16, 14*16
        # B, 3, 14*16, 14*16 -> B, 3, 14, 16, 14, 16
        # B, 3, 14, 16, 14, 16 -> B, 14, 14, 16, 16, 3
        #  B, 14*14, 16*16*3 - > B, num_patches, patch_dim
        
        #'b c (nh ph) (nw pw)': 
        # Đây là định dạng đầu vào của x mà chúng ta mong đợi. 
        # Nó ngụ ý rằng chiều cao h được chia thành nh "patch" theo chiều cao, 
        # mỗi patch có chiều cao ph. Tương tự, chiều rộng w được chia thành nw "patch" theo chiều rộng, 
        # mỗi patch có chiều rộng pw.
        
        #b (nh nw) (ph pw c): Đây là định dạng đầu ra mong muốn.
        
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                      ph=self.patch_height,
                      pw=self.patch_width)
        out = self.patch_embed(out)
        
        # Add cls
        #'d -> b 1 d': Biến đổi (D_model,) thành (batch_size, 1, D_model). 
        # 1 ở giữa là chiều của số lượng token (chỉ có 1 CLS token).
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=batch_size)
        out = torch.cat((cls_tokens, out), dim=1)
        
        # Add position embedding and do dropout
        out += self.pos_embed
        out = self.patch_emb_dropout(out)
        
        return out
    

# a = torch.tensor([[1]])
# b = torch.tensor([[2]])
# print(torch.cat((a, b), dim=0))  # Kết quả: tensor([1, 2])
