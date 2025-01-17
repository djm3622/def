import torch
import torch.optim as optim
from tqdm.notebook import tqdm
from wandb_helper import log_losses


def accelerator_train(accelerator, train, model, epochs, criterion, save_path, loss_type,
                     train_log, optimizer, scheduler, t_timesteps, config=None, loading_bar=False):
    
    # Cosine variance schedule as described in "Improved Denoising Diffusion Probabilistic Models"
    def cosine_beta_schedule(timesteps):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    beta = cosine_beta_schedule(t_timesteps)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0).to(accelerator.device)
    
    # TODO : finish the learned variance option
    
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    sqrt_alpha_bar= torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
    
    def step(train_batch, model, criterion):
        clean_images, noisy_images, rand_timesteps = train_batch
        alpha_bar_t = alpha_bar[rand_timesteps].view(-1, 1, 1, 1)
        
        next_state = (torch.sqrt(alpha_bar_t) * clean_images) + (torch.sqrt(1 - alpha_bar_t) * noisy_images)
        
        if loss_type == "simple":
            noise_pred = model(next_state, rand_timesteps.squeeze(-1), return_dict=False)[0]
        elif loss_type == "hybrid":
            model_output = model(next_state, rand_timesteps.squeeze(-1), return_dict=False)[0]
            noise_pred, var_pred = torch.chunk(model_output, 2, dim=1)
        
        # Simple loss (L_simple)
        simple_loss = criterion(noise_pred, noisy_images)

        if loss_type == "simple":
            return simple_loss
        
        elif loss_type == "hybrid":
            # VLB loss (L_vlb)
            # Calculate posterior mean and variance
            posterior_mean = (
                (sqrt_alpha_bar[t].reshape(-1, 1, 1, 1) * beta[t].reshape(-1, 1, 1, 1) * clean_images) +
                ((1 - sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)) * sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1) * var_pred)
            ) / (1 - sqrt_alpha_bar[t].reshape(-1, 1, 1, 1))
            
            posterior_variance = posterior_variance[t].reshape(-1, 1, 1, 1)
            
            # Calculate KL divergence
            vlb_loss = 0.5 * (-1.0 - torch.log(var_pred) + var_pred / posterior_variance + 
                            (posterior_mean - var_pred)**2 / posterior_variance)
            vlb_loss = vlb_loss.mean()
            
            # Combine losses with lambda = 0.001 as mentioned in the paper
            return simple_loss + 0.001 * vlb_loss
        
    
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
                optimizer.zero_grad()
                train_loss += loss.item()
            
            if loading_bar:
                loader.set_postfix(train_loss=loss.item())
                                            
        train_loss /= len(train)
        gathered_train_loss = accelerator.gather(torch.tensor([train_loss]).to(accelerator.device)).mean().item()
        train_log.append(gathered_train_loss)
        
        accelerator.print(f'Epoch {epoch+1}/{epochs}, Train Loss: {gathered_train_loss}')
        
        if accelerator.is_main_process:
            log_losses(
                train_loss=gathered_train_loss,
                valid_loss=None,
                step=epoch
            )
            
            scheduler.step()
            accelerator.save_model(model, save_path+'train')