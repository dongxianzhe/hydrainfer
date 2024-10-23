from typing import Union, Optional
import torch
from torch import Tensor, nn
# sample
# Args:
#     logits (Tensor): 
#         (num_tokens, vocab_size) float tensor, num_tokens is num_seqs if each seqs decode one token
#     unique_token_ids (Tensor): 
#         (num_tokens, max_seq_len) int tensor which each element is an int [0, vocab_size)  and the elements in one row are different from each other, because one row represent one seq and one seq has at most max_seq_len tokens thus one row at most has max_seq_len unique token_id. because all seq batch in one tensor so all the seq's number of unique tokens must pad to max_seq_len, each seq may has different number of unique_tokens, the number of unqiue tokens of seq i is unique_token_lens[i]
#         eg. in frequence and presence penalities the i's seq we will modify the logits[i, unique_token_lens[i,:]] with unique_token_counts[i, :] and 
#     unique_token_counts (Tensor): 
#         (num_tokens, max_seq_len) int tensor, used with unique_token_lens, 
#         unique_token_counts[i][j] represents unqiue_token_ids[i][j] appears unique_token_counts[i][j] times in seq i
#     unique_token_lens (Tensor):
#         (num_tokens) int Tensor, unique_token_lens[i] represents unique_token_ids[i]'s and unique_token_counts[i]'s length
#     frequency_penalties (Tensor):
#         (num_tokens, ) float tensor
#         each seq has one frequency_penalty param (-inf, +inf)
#         Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
#         0 will do nothing
#     presence_penalties (Tensor): 
#         (num_tokens, ) float tensor
#         each seq has one presence_penalty param  (-inf, +inf)
#         Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
#         0 will do nothing
#     repetition_penalties (Tensor):
#         (num_tokens, ) float tensor
#         each seq has one repetition_penalty param (0, +inf)
#         if repetion_penalty > 1, reduce the probility of repetition tokens
#         if repetion_penalty == 1, do nothing
#         if 0 < repetion_penalty < 1, increace the probility of repetition tokens
#     temperatures (Tensor): 
#         (num_tokens, ) float tensor
#         each seq has one temperature param [0, +inf)
#         if temperature > 1  do nothing
#         if temperature == 1 do nothing
#         if 0 < temperature < 1  do nothing
#         zero will do nothing
#     top_k (Tensor): 
#         (num_tokens, ) int tensor
#         each seq has one top_k param [0, +inf)
#         0 means do nothing
#         else select top k logit token
#     top_p (Tensor): 
#         (num_tokens, ) int tensor
#         each seq has one top_p param (-inf, +inf)
#         select sum probability greater than top_p

def sample(
    logits: Tensor, # (num_tokens, vocab_size)
    unique_token_ids: Tensor,  # (num_tokens, max_seq_len)
    unique_token_counts: Tensor, 
    unique_token_lens: Tensor,
    frequency_penalties: Tensor, # (num_tokens, )
    presence_penalties: Tensor,  # (num_tokens, )
    repetition_penalties: Tensor,
    temperatures: Tensor, # (num_tokens, )
    top_k: Tensor, 
    top_p: Tensor
) -> Tensor:
    # some mesc things
    num_tokens, vocab_size = logits.shape
    device=logits.device
    # 1. frequency and presence penalties
    score = logits.gather(dim=1, index=unique_token_ids) # score (num_tokens, max_seq_lens) scores[i][j] is logits[i][unique_token_ids[i][j]]
    score.sub_(unique_token_counts * frequency_penalties[:, None]) # broadcast (num_tokens, 1) to (num_tokens, max_seq_lens)
    score.sub_((unique_token_counts > 0) * presence_penalties[:, None])
    # 2. repetition penalties
    score = torch.where(score < 0, score * repetition_penalties[:, None], score / repetition_penalties[:, None])
    logits.scatter_(dim=1, index=unique_token_ids, src=score) # logits[i][unique_token_ids[i][j]] = score[i][j] for i in num_tokens, j in max_seq_lens
    # 3. temperatures 
    temperatures = torch.where(temperatures == 0, 1., temperatures) # temperatures[i][j] = temperatures[i][j] == 0 ? 1 : temperatures[i][j]
    logits.div_(temperatures[:, None])    
    # 4. topk
    top_k = torch.where(top_k <= 0, 2147483647, top_k)
    logits_sort, logits_ids = logits.sort(dim=-1, descending=True)
    top_k_mask = torch.arange(vocab_size, dtype=torch.int, device=device)[None, :] >= top_k[:, None]
    logits_sort.masked_fill_(mask=top_k_mask, value=float('-inf'))
    # 5. topp
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1) # probs_sum[i][j] means sum probs_soft[i][:j]
    top_p_mask = probs_sum - probs_sort > top_p[:, None]
    logits_sort.masked_fill_(mask=top_p_mask, value=float('-inf'))
    return logits_sort.gather(dim=-1, index=logits_ids.argsort())

def unique_randint(low, high, size):
    tensor = torch.empty(size)
    for i in range(size[0]):
        range_tensor = torch.arange(low, high)
        unique_tensor = range_tensor[torch.randperm(range_tensor.size(0))[:size[1]]]
        tensor[i] = unique_tensor
    return tensor