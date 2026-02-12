optimizer= torch.optim.AdamW(model.parameters(), lr= 5e-5)
for epoch in range(50):
  train_loss = 0.0
  model.train()
  for batch in tqdm(train_loader, desc= f"Epoch{epoch+1}"):
    src, dec_inp, lbls = [b.to(device) for b in batch]
    src_mask= (src != sp.pad_id()).to(device)
    dec_inp_mask= (dec_inp != sp.pad_id()).to(device)
    lbls_mask= (lbls != sp.pad_id()).to(device)
    optimizer.zero_grad()
    logits,loss = model(src, dec_inp, lbls)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
  train_loss /= len(train_loader)
  print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
  
