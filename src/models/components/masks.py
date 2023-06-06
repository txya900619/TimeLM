from typing import Optional, Tuple

import torch
from torch import Tensor


def get_key_padding_mask(padded_input: Tensor, pad_idx: int):
    """Creates a binary mask to prevent attention to padded locations.

    Arguments
    ----------
    padded_input: int
        Padded input.
    pad_idx:
        idx for padding element.

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_key_padding_mask(a, pad_idx=0)
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx).to(padded_input.device)

    # if the input is more than 2d, mask the locations where they are silence
    # across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()


def get_lookahead_mask(padded_input: Tensor):
    """Creates a binary mask for each sequence which maskes future frames.

    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device)) != 1).transpose(
        0, 1
    )
    return mask.detach().to(padded_input.device)


def make_masks(
    src: Tensor, pad_idx=0, look_ahead_mask=True, padding_mask=True
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    src_mask = None
    if look_ahead_mask:
        src_mask = get_lookahead_mask(src)

    src_key_padding_mask = None
    if padding_mask:
        src_key_padding_mask = get_key_padding_mask(src, pad_idx)

    return src_mask, src_key_padding_mask
