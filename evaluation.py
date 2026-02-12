model.eval()
  val_loss = 0.0
  with torch.no_grad():
    for batch in tqdm(val_loader, desc= "Validation"):
      src, dec_inp, lbls = [b.to(device) for b in batch]
      logits, loss = model(src, dec_inp, lbls)
      val_loss += loss.item()
    # Use the first validation sample as a test
    test_sentence = src_val_tensor[0].unsqueeze(0).to(device)
    # Start with the BOS token for the decoder
    generated = torch.tensor([[sp.bos_id()]], device= device, dtype= torch.long)
    for _ in range(block_size):
      decoder_input= generated
      decoder_input= decoder_input[:,-block_size:] if decoder_input.size(1) > block_size else decoder_input
      logits, _ = model(test_sentence, decoder_input, targets= None)
      next_token= logits.argmax(-1)[:,-1]
      if next_token.item() == sp.eos_id():
        break
      generated= torch.cat([generated,next_token.unsqueeze(1)], dim= 1)
    generated = generated[0].tolist()[1:]
     
    print('Generated Arabic text:', sp.DecodeIds(generated))
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/14:")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : avg_val_loss,
    },f'/content/checkpoints/epoch_{epoch +1}.pt')
