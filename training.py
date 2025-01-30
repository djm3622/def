import torch
import torch.optim as optim
from tqdm.notebook import tqdm
from wandb_helper import log_losses


@torch.no_grad()
def sampling(model, samples, t_timesteps, beta, alpha, alpha_bar, accelerator, size):
    c, w, h = 1, size, size

    # intial random noise like normal data
    imgs = torch.randn((samples, c, w, h), device=accelerator.device)

    for step in range(t_timesteps-1, -1, -1):
        # skip adding noise if timetep 0 (final step -> sample)
        error = torch.randn_like(imgs) if step > 1 else torch.zeros_like(imgs)

        # timesteps needed for the forward pass
        timesteps = torch.ones(samples, dtype=torch.int, device=accelerator.device) * step

        # change shape (batch_size) -> (batch_size, 1, 1, 1) to align w imgs
        beta_t = beta[timesteps].view(samples, 1, 1, 1)
        alpha_t = alpha[timesteps].view(samples, 1, 1, 1)
        alpha_bar_t = alpha_bar[timesteps].view(samples, 1, 1, 1)

        # formula
        mu = 1 / torch.sqrt(alpha_t) * (imgs - ((beta_t) / torch.sqrt(1 - alpha_bar_t)) * model(imgs, timesteps - 1, return_dict=False)[0])
        sigma = torch.sqrt(beta_t)
        imgs = mu + sigma * error

    return imgs


def accelerator_train(accelerator, train, model, epochs, criterion, save_path, loss_type, train_log,
                      optimizer, scheduler, sample_delay, t_timesteps, size, config=None, loading_bar=False):
    
    # Cosine variance schedule as described in "Improved Denoising Diffusion Probabilistic Models"
    def cosine_beta_schedule(timesteps):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    beta = torch.linspace(1e-4, 0.02, t_timesteps).to(accelerator.device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0).to(accelerator.device)
    
    # TODO : finish the learned variance option
    
    def step(train_batch, model, criterion):
        clean_images, noisy_images, rand_timesteps = train_batch
        alpha_bar_t = alpha_bar[rand_timesteps].view(-1, 1, 1, 1)
        
        next_state = (torch.sqrt(alpha_bar_t) * clean_images) + (torch.sqrt(1 - alpha_bar_t) * noisy_images)
        
        if loss_type == "simple":
            noise_pred = model(next_state, rand_timesteps.squeeze(-1), return_dict=False)[0]
        
        # Simple loss (L_simple)
        simple_loss = criterion(noise_pred, noisy_images)

        if loss_type == "simple":
            return simple_loss
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        
        if loading_bar:
            loader = tqdm(train, desc=f'Training', leave=False, mininterval=1.0)
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
        
        accelerator.print(f'Epoch {epoch+1}/{epochs}, Train Loss: {gathered_train_loss}')
        
        if accelerator.is_main_process:
            img = None
            
            if epoch % sample_delay == 0:
                img = sampling(model, 1, t_timesteps, beta, alpha, alpha_bar, accelerator, size)[0][0].cpu().numpy()
            
            log_losses(
                train_loss=gathered_train_loss,
                valid_loss=None,
                step=epoch,
                img=img
            )
            
            accelerator.save_model(model, save_path+'train')
            

