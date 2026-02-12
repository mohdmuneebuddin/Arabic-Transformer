class GPT(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        h = nn.ModuleList([BlockGPT(config)for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd)
    ))
    self.drop = nn.Dropout(config.drop_out)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
    self.lm_head.weight = self.transformer.wte.weight
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
      torch.nn.init.zeros_(module.bias)
  def forward(self, idx, encoder_output =None, targets=None, src_mask=None, dec_inp_mask= None):
    B,T = idx.shape
    device = idx.device
    pos = torch.arange(0,T, dtype = torch.long, device = device).unsqueeze(0)
    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = tok_emb + pos_emb
    x = self.drop(x)
    for block in self.transformer.h:
      x = block(x,encoder_output)

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index= sp.pad_id())
    return logits, loss

class BlockGPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.cross_attn = CrossAttention(config)
    self.ln3 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x, encoder_output):
    x = x + self.attn(self.ln1(x))
    x = x + self.cross_attn(self.ln3(x), encoder_output)
    x = x + self.mlp(self.ln3(x))
    return x

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_att = nn.Linear(config.n_embd, 3*config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.attn_dropout = nn.Dropout(config.drop_out)
    self.resid_dropout = nn.Dropout(config.drop_out)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.head_dim = config.head_dim
    self.register_buffer("bias",torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))
  def forward(self, x, dec_input_mask= None):
    B,T,C = x.shape
    q,k,v = self.c_att(x).split(self.n_embd, dim = 2)
    k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
    q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
    v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
    att = (q@k.transpose(-2,-1))/ ((self.head_dim)**0.5)
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    if dec_input_mask is not None:
      att = att.masked_fill(dec_input_mask[:,None,None,:] == 0, float('-inf'))
    att = F.softmax(att, dim = -1)
    att = self.attn_dropout(att)
    y = att@v
    y = y.transpose(1,2).contiguous().view(B,T,C)
    y = self.c_proj(y)
    y = self.resid_dropout(y)
    return y

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
    self.gelu = nn.GELU(approximate = 'tanh')
    self.dropout = nn.Dropout(config.drop_out())
    self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.dropout(x)
    x = self.c_proj(x)
    return x

class Encoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = nn.LayerNorm(config.n_embd)
    ))
    self.drop= nn.Dropout(config.drop_out)
    self.apply(self._init_weights)
  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      torch.nn.init.normal_(module.weight, mean= 0, std= 0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
      torch.nn.init.zeros_(module.bias)
  def forward(self, idx, src_mask=None):
    B,T = idx.shape
    device = idx.device
    pos = torch.arange(0,T, dtype = torch.long, device = device).unsqueeze(0)
    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = tok_emb + pos_emb
    x = self.drop(x)
    for block in self.transformer.h:
      x = block(x)
    x= self.transformer.ln_f(x)
    return x

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.attn = SelfAttention(config)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)
  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x

class SelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_att = nn.Linear(config.n_embd, 3*config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.attn_dropout = nn.Dropout(config.drop_out)
    self.resid_dropout = nn.Dropout(config.drop_out)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.head_dim = config.head_dim
  def forward(self, x, src_mask= None):
    B,T,C = x.shape
    qkv = self.c_att(x)
    q,k,v = qkv.split(self.n_embd, dim = 2)
    q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
    k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
    v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
    att = q@k.transpose(-2,-1)/ (self.head_dim**0.5)
    if src_mask is not None:
      att = att.masked_fill(src_mask[:,None,None,:] == 0, float('-inf'))
    att = F.softmax(att, dim= -1)
    att = self.attn_dropout(att)
    y = att@v
    y = y.transpose(1,2).contiguous().view(B,T,C)
    y = self.c_proj(y)
    y = self.resid_dropout(y)
    return y

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
    self.gelu = nn.GELU()
    self.dropout= nn.Dropout(config.drop_out)
    self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.dropout(x)
    x = self.c_proj(x)
    return x

class CrossAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.query = nn.Linear(config.n_embd, config.n_embd)
    self.key = nn.Linear(config.n_embd, config.n_embd)
    self.value = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.attn_dropout = nn.Dropout(config.drop_out)
    self.resid_dropout = nn.Dropout(config.drop_out)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.head_dim = config.head_dim
  def forward(self, x, encoder_output, src_mask= None, dec_input_mask= None):
    B,T_dec,C = x.shape
    B_enc, T_enc, C_enc = encoder_output.shape
    q = self.query(x)
    k = self.key(encoder_output)
    v = self.value(encoder_output)
    q = q.view(B,T_dec,self.n_head,C//self.n_head).transpose(1,2)
    k = k.view(B_enc,T_enc,self.n_head,C//self.n_head).transpose(1,2)
    v = v.view(B_enc,T_enc,self.n_head,C//self.n_head).transpose(1,2)
    att = q@k.transpose(-2,-1)/ ((self.head_dim)**0.5)
    if src_mask is not None:
      att = att.masked_fill(src_mask[:,None,None,:] == 0)
    att = F.softmax(att, dim = -1)
    att = self.attn_dropout(att)
    y = att@v
    y = y.transpose(1,2).contiguous().view(B,T_dec,C)
    y = self.c_proj(y)
    y= self.resid_dropout(y)
    return y

class Transformer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.encoder = Encoder(config)
    self.decoder = GPT(config)
  def forward(self, source_id, idx, targets=None, src_mask=None, dec_inp_mask=None):
    encoder_output = self.encoder(source_id, src_mask=None)
    logits, loss = self.decoder(idx, encoder_output, targets, dec_inp_mask=None)
    return logits, loss
model = Transformer(config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
