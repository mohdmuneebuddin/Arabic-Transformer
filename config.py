class GPTConfig:
  def __init__(self,
                block_size: int = block_size,
                vocab_size= Vocab_size,
                n_head= 8,
                n_layer= 6,
                n_embd= 512,
                drop_out= 0.1,):
    self.block_size = block_size
    self.vocab_size = vocab_size
    self.n_head = n_head
    self.n_layer = n_layer
    self.head_dim = n_embd // n_head
    self.n_embd = n_embd
    self.drop_out = drop_out

config= GPTConfig()
