import torch

def stat_model_param_number(model: torch.nn.Module):
    num_of_params = 0
    param_dict = {}
    for name,par in model.named_parameters():
        # only include trainable parameters
        if par.requires_grad:
            num_of_params += par.numel()
            param_dict[name] = par.numel()
    print('params:', num_of_params)
    for k,v in param_dict.items():
        print(k, v)