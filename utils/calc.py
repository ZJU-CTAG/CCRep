import torch

# todo: miss handling of zero-division problem
def nanmean(x: torch.Tensor, dim = None) -> torch.Tensor:
    nan_mask = torch.isnan(x)
    non_nan_mask= ~nan_mask

    if dim is not None:
        non_nan_count = non_nan_mask.sum(dim=dim)
    else:
        non_nan_count = non_nan_mask.sum()

    # # Handle zero-division problem when all items are masked
    non_nan_count = non_nan_count.masked_fill(non_nan_count==0, 1)

    if dim is not None:
        nan_sum = torch.nansum(x, dim)
    else:
        nan_sum = torch.nansum(x)

    return nan_sum / non_nan_count