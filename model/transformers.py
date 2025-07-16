import torch.nn as nn
from patch_embedding import PatchEmbedding
from transformers_layer import TransformerLayer
import torch

class VIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers = config['n_layers']
        emb_dim = config['emb_dim']
        num_classes = config['num_classes']
        self.patch_embed_layer = PatchEmbedding(config)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(n_layers) #số lượng stack transformer block chồng lên nhau
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.fc_number = nn.Linear(emb_dim, num_classes)
        
    def forward(self, x):
        # Patchify and add CLS token
        out = self.patch_embed_layer(x)
        
        # Go through the transformer layers
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        
        # Compute logits
        return self.fc_number(out[:, 0]) #0 for special cls token, the first dimension (:) is the batch dimension, cls token is the first row
    
    
a = torch.randn(3, 3, 3)  # Example input: batch size of 2, 3 channels, 224x224 image
print(a)
print(a[:, 0])
