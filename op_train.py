import torch
import torch.optim as optim
from tqdm.notebook import tqdm
from wandb_helper import log_losses

def check_point_accelerate(
    val_loss, best_val_loss, model, patience_counter, 
    save_path, accelerator, optimizer, epoch
):
    gathered_val_loss = accelerator.gather(torch.tensor([val_loss]).to(accelerator.device)).mean().item()
    
    if gathered_val_loss < best_val_loss:
        best_val_loss = gathered_val_loss
        patience_counter = 0
        accelerator.save_model(model, save_path+'valid') 
        return patience_counter, best_val_loss
    else:
        patience_counter += 1
    return patience_counter, best_val_loss


def step(batch, model, criterion):
    x, y, t = batch
    out = model(x, t.squeeze(-1), return_dict=False)[0]
    loss = criterion(out, y)

    return loss
    

def accelerator_train_operator(
    accelerator, train, valid, model, epochs, patience, criterion, save_path, 
    train_log, valid_log, optimizer, scheduler, config=None, loading_bar=False, val_delay=1
):    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        if loading_bar:
            loader = tqdm(train, desc=f'Training', leave=False, mininterval=20.0)
        else:
            loader = train
        
        for batch_idx, train_batch in enumerate(loader):
            with accelerator.accumulate(model):
                loss = step(train_batch, model, criterion)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            
            if loading_bar:
                loader.set_postfix(train_loss=loss.item())
                                            
        train_loss /= len(train)
        gathered_train_loss = accelerator.gather(torch.tensor([train_loss]).to(accelerator.device)).mean().item()
        train_log.append(gathered_train_loss)
        
        if epoch % val_delay != 0:
            accelerator.print(f'Epoch {epoch+1}/{epochs}, Train Loss: {gathered_train_loss}')
            
            # Log just train loss
            if accelerator.is_main_process:
                log_losses(
                    train_loss=gathered_train_loss,
                    valid_loss=None,
                    step=epoch
                )
            
            valid_log.append(0)
            accelerator.save_model(model, save_path+'train')
            continue
            
        model.eval()
        val_loss = 0
        if loading_bar:
            loader = tqdm(valid, desc=f'Validation', leave=False, mininterval=20.0)
        else:
            loader = valid
        
        with torch.no_grad():
            for valid_batch in loader:
                loss = step(valid_batch, model, criterion)
                val_loss += loss.item()
                
                if loading_bar:
                    loader.set_postfix(val_loss=loss.item())
                                    
        scheduler.step()
        
        val_loss /= len(valid)
        accelerator.wait_for_everyone()
        gathered_val_loss = accelerator.gather(torch.tensor([val_loss]).to(accelerator.device)).mean().item()
        valid_log.append(gathered_val_loss)
        
        # Log epoch metrics to wandb if main process
        if accelerator.is_main_process:
            log_losses(
                train_loss=gathered_train_loss,
                valid_loss=gathered_val_loss,
                epoch=epoch
            )
        
        accelerator.print(f'Epoch {epoch+1}/{epochs}, Train Loss: {gathered_train_loss}, Validation Loss: {gathered_val_loss}')
        
        patience_counter, best_val_loss = check_point_accelerate(
            val_loss, best_val_loss, model, patience_counter, save_path, 
            accelerator, optimizer, epoch
        )
        
        if patience_counter >= patience:
            accelerator.wait_for_everyone()
            accelerator.print('Early stopping triggered')
            break