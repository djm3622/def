import torch
import numpy as np


def indepent_channel_select(x_vec, channels=[3], seq_len=1):  
    tensor = torch.zeros((len(channels)*seq_len,) + x_vec.shape[1:])
    
    for ind in range(x_vec.shape[0] // 4):
        base_idx = ind * 4
        for idx, ch in enumerate(channels):
            if ch == 1:
                tensor[idx+ind*len(channels)] = x_vec[base_idx+1] / x_vec[base_idx]
            if ch == 2:
                tensor[idx+ind*len(channels)] = x_vec[base_idx+2] / x_vec[base_idx]
            if ch == 3:
                tensor[idx+ind*len(channels)] = x_vec[base_idx+3] / x_vec[base_idx]
    
    return tensor


def load_instances(data_instance, upsample_size=96, dtype=torch.float32, channels=4, seq_len=1, channel_select=[3]):
    np_instances = np.zeros((1, channels*10000))
    
    np_instances[0, :] = np.load(data_instance)
    
    torch_instances = torch.from_numpy(np_instances).to(dtype=dtype)
    
    if upsample_size is not None:
        inpt = nn.functional.interpolate(torch_instances.view(1, seq_len*channels, 100, 100), size=(upsample_size, upsample_size), mode='bicubic')
        return indepent_channel_select(inpt.view(channels*seq_len, upsample_size, upsample_size), seq_len=seq_len, channels=channel_select)
    
    return indepent_channel_select(inpt.view(channels*seq_len, 100, 100), seq_len=seq_len, channels=channel_select)  


def preprocess(input_tensor, target_tensor):
    # Compute mean and std for each image in the batch
    # keepdim=True maintains the dimension for broadcasting
    in_means = torch.mean(input_tensor, dim=(1, 2), keepdim=True)
    in_stds = torch.std(input_tensor, dim=(1, 2), keepdim=True)
    
    tar_means = torch.mean(target_tensor, dim=(1, 2), keepdim=True)
    tar_stds = torch.std(target_tensor, dim=(1, 2), keepdim=True)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    
    # Apply standardization: (x - mean) / std
    standard_input = (input_tensor - in_means) / (in_stds + eps)
    standard_target = (target_tensor - tar_means) / (tar_stds + eps)
    
    return standard_input, standard_target


def pre_step(out):    
    mu = out.mean(dim=(-2, -1), keepdims=True)
    std = out.std(dim=(-2, -1), keepdims=True)
    eps = 1e-8
    
    return (out - mu) / (std + eps)