import torch

# Function to sum across top-k groups
def sum_out(input: torch.Tensor, output: torch.Tensor) -> None:
    """
    Sums input tensor across top-k groups and stores the result in the output tensor.
    
    Parameters:
    input (torch.Tensor): A tensor of shape [n_tokens, topk, dim]
    output (torch.Tensor): A tensor of shape [n_tokens, dim] that will hold the output sum.
    
    Returns:
    None
    """
    ...


# Function to perform grouped top-k sigmoid
def grouped_topk_sigmoid(
    gating_logits: torch.Tensor,    # Shape: [n_tokens, n_experts]
    correction_bias: torch.Tensor,  # Shape: [n_experts]
    n_expert_groups: int,
    topk_group: int,
    topk: int,
    scaling_factor: float,
    topk_weights: torch.Tensor,     # Shape: [n_tokens, topk]
    topk_indices: torch.Tensor      # Shape: [n_tokens, topk]
) -> None:
    """
    Computes grouped top-k sigmoid values.

    Parameters:
    gating_logits (torch.Tensor): Tensor of shape [n_tokens, n_experts]
    correction_bias (torch.Tensor): Correction bias, shape [n_experts]
    n_expert_groups (int): Number of expert groups
    topk_group (int): Top-K group
    topk (int): The top K values to select
    scaling_factor (float): A factor used to scale the logits
    topk_weights (torch.Tensor): A tensor to store top-k weights, shape [n_tokens, topk]
    topk_indices (torch.Tensor): A tensor to store top-k indices, shape [n_tokens, topk]

    Returns:
    None
    """
    ...


def permute_with_index_map(
    tokens: torch.Tensor,   # Shape: [n_tokens, dim]
    topk_ids: torch.Tensor  # Shape: [n_tokens, topk]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Permutes the tokens based on the index map (topk_ids).

    Parameters:
    tokens (torch.Tensor): A tensor of shape [n_tokens, dim].
    topk_ids (torch.Tensor): A tensor of shape [n_tokens, topk], which specifies the top-k indices.

    Returns:
    tuple: A tuple containing two tensors:
        - The permuted tokens tensor of shape [n_tokens, dim].
        - The associated permuted indices tensor of shape [n_tokens, topk].
    """
    ...


# Function to unpermute a tensor based on an index map
def unpermute_with_index_map(
    permuted_tokens: torch.Tensor,  # Shape: [n_permuted_tokens, dim]
    row_id_map: torch.Tensor,       # Shape: [topk, n_tokens] => dst row
    probs: torch.Tensor             # Shape: [n_tokens, topk]
) -> torch.Tensor:
    """
    Unpermuted the input tensor based on the row_id_map and the probabilities.

    Parameters:
    permuted_tokens (torch.Tensor): Tensor of shape [n_permuted_tokens, dim]
    row_id_map (torch.Tensor): The index map, shape [topk, n_tokens]
    probs (torch.Tensor): Probabilities, shape [n_tokens, topk]

    Returns:
    torch.Tensor: The unpermuted tensor.
    """
    ...


def permute_with_mask_map(
    tokens: torch.Tensor,       # Shape: [n_tokens, dim]
    routing_map: torch.Tensor,  # Shape: [n_tokens, n_experts], a boolean tensor
    topk: int                   # The top-k value
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Permutes the tokens based on the mask map (routing_map).

    Parameters:
    tokens (torch.Tensor): A tensor of shape [n_tokens, dim].
    routing_map (torch.Tensor): A boolean tensor of shape [n_tokens, n_experts], indicating which experts are selected.
    topk (int): The number of top experts to consider for permutation.

    Returns:
    tuple: A tuple containing two tensors:
        - The permuted tokens tensor of shape [n_tokens, dim].
        - The associated permuted mask tensor of shape [n_tokens, topk].
    """
    pass


# Function to unpermute a tensor based on a mask map
def unpermute_with_mask_map(
    permuted_tokens: torch.Tensor,  # Shape: [n_permuted_tokens, dim]
    row_id_map: torch.Tensor,       # Shape: [n_experts, n_tokens] => dst row
    probs: torch.Tensor             # Shape: [n_tokens, n_experts]
) -> torch.Tensor:
    """
    Unpermutes the input tensor based on the mask map and probabilities.

    Parameters:
    permuted_tokens (torch.Tensor): Tensor of shape [n_permuted_tokens, dim]
    row_id_map (torch.Tensor): The mask map, shape [n_experts, n_tokens]
    probs (torch.Tensor): Probabilities, shape [n_tokens, n_experts]

    Returns:
    torch.Tensor: The unpermuted tensor.
    """
    ...


# Function to compute top-k softmax values
def topk_softmax(
    gating_logits: torch.Tensor,  # Shape: [n_tokens, n_experts]
    topk_weights: torch.Tensor,   # Shape: [n_tokens, topk]
    topk_indices: torch.Tensor    # Shape: [n_tokens, topk]
) -> None:
    """
    Computes top-k softmax values and stores them in topk_weights and topk_indices.

    Parameters:
    gating_logits (torch.Tensor): Tensor of shape [n_tokens, n_experts]
    topk_weights (torch.Tensor): Tensor of shape [n_tokens, topk] to store the top-k weights
    topk_indices (torch.Tensor): Tensor of shape [n_tokens, topk] to store the top-k indices

    Returns:
    None
    """
    ...
