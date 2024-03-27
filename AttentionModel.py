# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:19:30 2024

@author: Mels
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from layers import MultiHeadAttention, FeedFoward

def save_model(model):
    torch.save(model, 'Results/Model')

def load_model():
    model = torch.load('Results/Model')
    model.eval()
    return model  


#%%
class Block(nn.Module):
    """
    Transformer block: This class contains the MultiHeadAttention Mechanism together
    with a FeedForward layer. In between the layers are normalised and residue is added.
    The normalisation happens to make the hyperparameters easier to compute. Residue is 
    added such that the gradient can be calculated backwards, making training easier.    
    """

    def __init__(self, n_heads, n_embd, block_size, dropout):
        # n_embd: embedding dimension, n_heads: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads=n_heads, n_embd=n_embd, head_size=head_size, block_size=block_size, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class AttentionModel(nn.Module):
    """
    Adds together the different transformer blocks, together with some other important layers:
        - embedding
        - positional embedding (to encode the position of the logits into the embedding, 
                                otherwise information is lost in the attention mechanism)
        - Blocks: The transformer block that contains 
            - MultiHeadAttention
            - FeedForward
            - Layernorm
            - Residue
            (heavily inspired by the Attention is All You Need paper)
        - layernorm
        - linear layer
    """
    
    def __init__(self, vocab_size : int, n_layer : int, n_heads : int, n_embd : int, block_size : int, dropout : float):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_heads=n_heads, n_embd=n_embd, 
                                            block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.block_size = block_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


