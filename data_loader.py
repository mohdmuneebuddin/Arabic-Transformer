spm.SentencePieceTrainer.train(f'--input={txt_file_out} --model_prefix=spm_mt --vocab_size={Vocab_size} --model_type=bpe --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3')
sp = spm.SentencePieceProcessor(model_file = 'spm_mt.model')
source_id = [[sp.bos_id()] + sp.EncodeAsIds(s) + [sp.eos_id()] for s in df['english']]
target_id = [[sp.bos_id()] + sp.EncodeAsIds(t) + [sp.eos_id()] for t in df['arabic']]
src_train, src_val, tgt_train, tgt_val = train_test_split(source_id, target_id, test_size = 0.1, random_state = 42)
def pad_sequences(sequences, max_length, pad_value= 0):
  return torch.tensor([seq + [pad_value] * (max_length - len(seq)) if len(seq)< max_length else seq[:max_length] for seq in sequences])

src_train_tensor = pad_sequences(src_train, block_size)
src_val_tensor = pad_sequences(src_val, block_size)
tgt_train_tensor = pad_sequences(tgt_train, block_size)
tgt_val_tensor = pad_sequences(tgt_val, block_size)

src_train_mask = (src_train_tensor != sp.pad_id())

decoder_input_train = tgt_train_tensor[:,:-1]
decoder_label_train = tgt_train_tensor[:,1:]
decoder_input_val = tgt_val_tensor[:,:-1]
decoder_label_val = tgt_val_tensor[:,1:]

train_dataset = TensorDataset(
                              src_train_tensor,
                              decoder_input_train,
                              decoder_label_train
                              )
train_loader = DataLoader(
                          train_dataset,
                          batch_size= batch_size,
                          shuffle = True
                          )
val_dataset = TensorDataset(
                            src_val_tensor,
                            decoder_input_val,
                            decoder_label_val
)
val_loader = DataLoader(
                        val_dataset,
                        batch_size = batch_size,
                        shuffle = False
                        )
