import torch
import numpy as np
import utility
import samplers
import deterministic


# meta data comes in under format: 
# [(start, stop, spaced), (start, stop, spaced), ...]
# todo figure out how to automate using the time_embedding dataset
def spaced_instances(instance_meta):
    pass


# sample_size = len(ground)-1
# pred_num = # of samples
# set imgs to None if using multistep for correct init
# states should already be perturbed before coming in
@torch.no_grad()
def ensemble_pred(
    operator, device, states, sampling_func, sample_params, pred_len, 
    pred_num, multi_step_sampling=False, conditions=None, multi_stop=0
):
    operator.eval()
    state_size = states.shape[-1] if states is not None else conditions.shape[-1] 
    history_pred = torch.zeros(pred_len, pred_num, 1, state_size, state_size)
    
    if not multi_step_sampling:
        out = operator(states.to(device), torch.ones(samples).to(device))[0]
    else:
        out = operator(conditions.to(device), torch.ones(samples).to(device))[0]
        
    out = deterministic.pre_step(out)
    
    history_pred[0] = out.cpu()

    if multi_step_sampling:
        out, mixed_condition = sampling_func(*sample_params)

    for k in range(pred_len-1):
        out = operator(out, torch.ones(samples).to(device))[0]
        out = pre_step(out)

        if multi_stop < 5 and multi_step_sampling:
            out, mixed_condition = sampling_func(*sample_params)

        history_pred[k+1] = out.cpu()
    
    return history_pred


# the state should be size [1, 1, 96, 96] before coming in
@torch.no_grad()
def deterministic_pred(
    operator, device, state, pred_len
):
    operator.eval()
    state_size = state.shape[-1]
    history_pred = torch.zeros(pred_len, 1, 1, state_size, state_size)
    
    out = operator(state.to(device), torch.ones(1).to(device))[0]
    out = deterministic.pre_step(out)
    
    history_pred[0] = out.cpu()
    
    for k in range(len(ground)-2):
        out = operator(out, torch.ones(1).to(device))[0]
        out = deterministic.pre_step(out)
        
        history_pred[k+1] = out.cpu()

    return history_pred



def tune(solver):
    pass