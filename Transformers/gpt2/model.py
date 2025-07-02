import torch
from dataclasses import dataclass
from torch import nn
from transformers import AutoTokenizer

@dataclass
class config: 
    vocab_size: int = 50257
    embedding_dim: int = 768
    num_attention_heads: int = 6
    num_attention_blocks: int = 6
    ff_hidden_dim: int = 4*768
    max_seq_len = 128
    dropout=0.1
    device='cuda' if torch.cuda.is_available() else 'cpu'
    bias: bool = True
    

class CausalAttentionHead(nn.Module):
    def __init__(self, config: config):
        super().__init__()
        
        self.embedding_dim = config.embedding_dim
        self.head_size = self.embedding_dim // config.num_attention_heads
        
        # There are four matrices W_q, W_k, W_v, W_o
        # head_size, embedding_dim
        self.device = torch.device(config.device)
        self.W_q = nn.Parameter(torch.zeros(self.head_size, self.embedding_dim))
        self.W_k = nn.Parameter(torch.zeros(self.head_size, self.embedding_dim))
        self.W_v = nn.Parameter(torch.zeros(self.head_size, self.embedding_dim))
        
        torch.nn.init.normal_(self.W_q, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.W_k, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.W_v, mean=0.0, std=0.02)
            
    def forward(self, X, padding_mask):
        #X: batch, seq, features 
        #padding: batch, seq
        
        #we needs to make it (batch, seq, 1) <- this allows broadcasting along dim=2
        padding_mask = padding_mask.unsqueeze(2)
        X = X * padding_mask
        
        seq_len = X.shape[1]
        
        #: batch, seq, head_size
        X_q = X @ self.W_q.T
        X_k = X @ self.W_k.T
        X_v = X @ self.W_v.T
        
        causal_attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).unsqueeze(0)
        
        
        #Each element in the row i represents how much of key k_j (j in head_size) is similar??? to query v_i
        scaled_attention_scores = torch.bmm(X_q, X_k.transpose(2,1)) / (self.head_size ** 0.5) # batch, seq, seq
        attention = torch.softmax(scaled_attention_scores.masked_fill(causal_attention_mask==0, float('-inf')), dim=2) # batch, seq, seq
        attention = torch.bmm(attention, X_v) # batch, seq, head_size
        
        return attention

class SelfAttention(nn.Module):
    def __init__(self, config: config):
        super().__init__()
        
        self.device = torch.device(config.device)
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.head_size = self.embedding_dim // config.num_attention_heads
        
  
        self.attention_heads = nn.ModuleList([
            CausalAttentionHead(config) 
            for _ in range(self.num_heads)
        ])
        
        self.W_o = nn.Linear(self.embedding_dim, self.embedding_dim) 
        
        
    def forward(self, X, padding_mask):
        #Each element: batch, seq, head_size
        head_outputs = []
        for head in self.attention_heads:
            head_outputs.append(head(X, padding_mask))
        
        # Concatenate all head outputs
        #batch, seq, embedding_dim
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # Apply output projection
        output = self.W_o(concatenated)
        
        return output
    

class TransformerBlock(nn.Module):
    def __init__(self, config:config):
        super().__init__()
        
        self.device = torch.device(config.device)
        self.attention_block = SelfAttention(config)
        self.layerNorm = nn.LayerNorm(config.embedding_dim, bias=config.bias)
        
        self.ff_hidden_dim = config.ff_hidden_dim
        self.linear = nn.Sequential(
            nn.Linear(config.embedding_dim, self.ff_hidden_dim, bias=config.bias), #bias = True
            nn.GELU(),
            nn.Linear(self.ff_hidden_dim, config.embedding_dim, bias=config.bias), #bias = True
            nn.GELU()
        )
        
    def forward(self, X, padding_mask):
        #X: batch, seq, features 
        
        SelfAttention_out = self.layerNorm(X + self.attention_block(X, padding_mask))
        linear_out = self.layerNorm(SelfAttention_out + self.linear(SelfAttention_out))
        
        return linear_out



class GPT(nn.Module):
        def __init__(self, config: config):
            super().__init__() 
            
            self.config = config
            self.device = torch.device(config.device)
            
            self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.embedding_dim)
            
            self.drop = nn.Dropout(config.dropout)
              
            #batch, seq, embedding_dim
            self.transformer = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_attention_blocks)])
        
            #embedding_dim, vocab_size -> batch, seq, vocab_size
            self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
            
            self.apply(self._init_weights)
            self.to(self.device)
        
        
        def forward(self, X, padding_mask):
            # X: batch, seq
            
            # Move inputs to the configured device
            X = X.to(self.device)
            padding_mask = padding_mask.to(self.device)
            
            batch_size = X.shape[0]
            seq_len = X.shape[1]
            
            #batch, seq, embedding_dim
            token_embedding = self.token_embedding(X)
            
            #1, seq_len
            positions = torch.arange(seq_len, device=self.device)
            #seq_len -> 1, seq -> batch, seq 
            positions = positions.unsqueeze(0).expand(batch_size, seq_len)
            
            #batch, seq, embedding_dim
            position_embedding = self.token_embedding(positions)
            
            X = self.drop(token_embedding + position_embedding)
            
            for block in self.transformer:
                X = block(X, padding_mask)
            
            out = self.lm_head(X)
            
            return out
        

        def generate(self, tokenized_prompt, padding_mask, new_tokens: int):
            #batch, seq -> batch, seq, vocab
            
            for x in range(0, new_tokens):
                out =  self.forward(tokenized_prompt, padding_mask)
                
                last_token: torch.Tensor = out[:,  -1, :] #batch, vocab_size
                                
                probs = torch.softmax(last_token, dim = 1).to(config.device) 
                token_ids = probs.argmax(dim = 1).unsqueeze(1) #batch, 1
                
                tokenized_prompt = torch.cat([tokenized_prompt, token_ids], dim=1)
                
            
            
        
        def _init_weights(self, module):
            if isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                
        
