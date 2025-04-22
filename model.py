import torch
import torch.nn as nn
import math


# INPUT EMBEDDING
class InputEmbedding(nn.Module):
	def __init__ (self, d_model:int, vocab_size: int):
		super().__init__()
		self.d_model = d_model
		self.vocab_size = vocab_size
		self.embedding = nn.Embedding(vocab_size,d_model)
	
	# 3.4 Embeddings and Softmax 
	def forward(self, x):
		return self.embedding(x) * math.sqrt(self.d_model) 

# POISITIONAL ENCODING
class PositionalEncoding(nn.Module):
	
	def __init__(self, d_model: int, seq_len:int , dropout: float)->None:
		super().__init__()
		self.d_model = d_model
		self.seq_len = seq_len
		self.dropout = nn.Dropout(dropout)
		
		# Create a matrix of shape (seq_len, d_model)
		pe = torch.zeros(seq_len, d_model)
		# Create a vector of shape (seq_len, 1)
		position =  torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
		# Apply th sin to pair
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0) # (1, seqlen, dmodel)

		self.register_buffer('pe', pe) # to save the tensor and the state of the model

	def forward(self, x):
		x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
		return self.dropout(x)

# ADD AND NORM LAYER
class LayerNormalization(nn.Module):
	
	def __init__ (self, eps: float=10**-6) -> None:
		super().__init__()
		self.eps = eps # for numerical stability and avoid div by 0 
		self.alpha = nn.Parameter(torch.ones(1)) # multiplied 
		self.bias = nn.Parameter(torch.zeros(1)) # added
		
	def forward(self, x):
		mean = x.mean(dim = -1, keepdim = True)
		std = x.std(dim = -1, keepdim = True )
		return self.alpha * (x - mean) / (std + self.eps) + self.bias
		

# FEED FORWARD
# 3.3 Position-wise Feed-Forward Networks
class FeedForwardBlock(nn.Module):
	
	def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
		super().__init__()
		self.linear_1 = nn.Linear(d_model,d_ff) # W1 and B1
		self.dropout = nn.Dropout(dropout)
		self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2
		
	def forward(self ,x):
		#  (Batch, SeqLen, d_model) -> (Batch, SeqLen, d_ff) -> (Batch, SeqLen, d_model)
		return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
	

# MULTIHEAD ATTENTION
class MultiHeadAttentionBlock(nn.Module):
	
	def __init__(self, d_model:int, h:int, dropout: float) -> None:
		super().__init__()
		self.d_model = d_model
		self.h = h
		assert d_model % h == 0, "d_model is not divisible by h"
		
		self.d_k = d_model // h
		self.w_q = nn.Linear(d_model,d_model) #Wq
		self.w_k = nn.Linear(d_model,d_model) #Wk
		self.w_v = nn.Linear(d_model,d_model) #Wv
		
		self.w_o = nn.Linear(d_model,d_model) #Wo
		self.dropout = nn.Dropout(dropout)
  
	@staticmethod
	def attention(query, key, value, mask, dropout: nn.Dropout):
		d_k = query.shape[-1]

		# (Batch, h, SeqLen, d_k) -> (Batch, h, SeqLen, SeqLen)
		attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
		if mask is not None:
			attention_scores.masked_fill_(mask == 0, -1e9)
		attention_scores = attention_scores.softmax(dim= -1) # (Batch, h, SeqLen, SeqLen)
		if dropout is not None:
			attention_scores = dropout(attention_scores)
   
		# used for attention, used for visualisation
		return (attention_scores @ value), attention_scores
  
  
	def forward(self, q, k, v, mask):
		query = self.w_q(q) # (Batch, SeqLen, d_model) -> (Batch, SeqLen, d_model)
		key = self.w_k(k) # (Batch, SeqLen, d_model) -> (Batch, SeqLen, d_model)
		value = self.w_v(v) # (Batch, SeqLen, d_model) -> (Batch, SeqLen, d_model)
		
		# (Batch, SeqLen, d_model) -> (Batch, SeqLen, h, d_k) -> (Batch, h, SeqLen, d_k)
		query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
		key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
		value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

		x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

		# (Batch, h, SeqLen, d_k) -> (Batch, SeqLen, h, d_k) -> (Batch, SeqLen, d_model)
		x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
		
		#  (Batch, SeqLen, d_model) -> (Batch, SeqLen, d_model)
		return self.w_o(x)