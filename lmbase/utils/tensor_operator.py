"""
Useful functions to process the tensors either the token ids or the general ones.
"""

from typing import Tuple, List, Union

import torch


def get_target_indices(
    input_ids: torch.Tensor,
    start_flag_ids: Tuple[torch.Tensor, List[int]],
    end_flag_id: Tuple[torch.Tensor, int],
    is_return_content: bool = False,
) -> Union[
    Tuple[List[torch.Tensor], List[torch.Tensor]], Tuple[List[torch.Tensor], None]
]:
    """
    Extract the desire content and the corresponding positions from the
    `input_ids` based on the given flags.

    :param input_ids: Contain the ids to be searched
     Shape (batch_size, L)
    :param start_flag_ids: Contain the ids behaving as the start indicator
     of the target content.
     Shape (l,)
    :param end_flag_ids: Contain the ids behaving as the end indicator of the
     target content.
     Shape (l,)
    :param is_return_content: whether return the extract content instead of only
     the indices

    """
    # Convert response_flag_ids to tensor if it's a list
    if isinstance(start_flag_ids, list):
        start_flag_ids = torch.tensor(start_flag_ids, device=input_ids.device)
    else:
        start_flag_ids = input_ids.to(input_ids.device)

    # Ensure end_id is a scalar tensor
    if isinstance(end_flag_id, torch.Tensor):
        end_flag_id = end_flag_id.item()

    # Get the flag_len
    l = len(start_flag_ids)

    # Flat the input_ids
    # Get the (batch_size, length)
    B, _ = input_ids.shape
    # input_ids = input_ids.view(-1)

    target_contents = []
    target_indices = []
    for batch_idx in range(B):
        batch_ids = input_ids[batch_idx]
        # Create sliding windows of length `l` from input_ids
        windows = batch_ids.unfold(0, l, 1)
        # Check which windows match B
        matches = (windows == start_flag_ids).all(dim=1)
        # Get starting indices of matches
        # Shape (#number of matched start_flag_ids, )
        match_indices = torch.nonzero(matches).squeeze()
        # Compute starting indices of the start_flag_ids
        # Shape (#number of the matches flags, )
        start_indices = match_indices + l

        # Get the end flag indices
        all_end_indices = torch.where(batch_ids == end_flag_id)[0]

        # Find the insertion indices for each value in A such that the insertion
        # would maintain the order. Using `right=True` ensures we get the next higher element.
        sort_indices = torch.searchsorted(all_end_indices, start_indices, right=True)

        # Use the indices to index C, which gives the minimal value in C that is larger than each value in A.
        end_indices = all_end_indices[sort_indices]

        # Get the indices
        # Make the shape to be (1, #number of...) for the stack usage
        if start_indices.dim() == 0:
            start_indices = start_indices.unsqueeze(0)
            end_indices = end_indices.unsqueeze(0)
        batch_indices = torch.stack([start_indices, end_indices], dim=1)

        slices = [batch_ids[start:end] for start, end in batch_indices]

        target_contents.append(slices)
        target_indices.append(batch_indices)

    if is_return_content:
        return target_indices, target_contents
    else:
        return target_indices, None


def find_tensor(src_tensor: torch.Tensor, tgt_values: List[List[int]]):
    """Find the tgt_tensor in a large tensor.

    :param src_tensor: Contain the tensor to be searched, in Shape
     (batch_size, L)
    :param tgt_tensor: Contain the target values, each item is a list
     containing the values to search.
    """
    batch_size = src_tensor.shape[0]
    # Reinitialize indices
    indices = torch.full((batch_size, 2), -1, dtype=torch.long)

    # Iterate over the batch dimension
    for i in range(batch_size):
        # Convert B[i] to a tensor
        values = torch.tensor(tgt_values[i], dtype=src_tensor.dtype)
        l = values.size(0)  # Length of B[i]
        L = src_tensor.size(1)  # Length of A[i]

        # Create sliding windows for A[i]
        windows = src_tensor[i].unfold(0, l, 1)
        # Compare each window to B[i]
        matches = (windows == values).all(dim=1)
        # Find the first matching index
        match_indices = torch.nonzero(matches, as_tuple=True)[0]
        if match_indices.numel() > 0:
            start_index = match_indices[0].item()  # Starting index
            end_index = start_index + l - 1  # Ending index
            indices[i] = torch.tensor([start_index, end_index])

    return indices
