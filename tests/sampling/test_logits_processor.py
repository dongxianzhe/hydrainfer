import torch
from dxz.sampling.logits_processor import process_logits

def unique_randint(low, high, size):
    tensor = torch.empty(size, dtype=torch.int)
    for i in range(size[0]):
        range_tensor = torch.arange(low, high)
        unique_tensor = range_tensor[torch.randperm(range_tensor.size(0))[:size[1]]]
        tensor[i] = unique_tensor
    return tensor

if __name__ == '__main__':
    batch_size = num_tokens = 4
    vocab_size = 32000
    max_seq_len = 1023

    logits = torch.randn(size=(batch_size, vocab_size))
    unique_token_ids = unique_randint(low = 1, high = vocab_size, size=(batch_size, max_seq_len))
    unique_token_counts = torch.randint(low=1, high=3, size=(batch_size, max_seq_len))
    unique_token_lens = torch.randint(low=1, high=max_seq_len, size=(batch_size, ))
    frequency_penalties = torch.tensor([0.01, 0.02, 0.03, 0.04])
    presence_penalties = torch.tensor([0.1, 0.2, 0.3, 0.4])
    repetition_penalties = torch.tensor([1.0, 2.0, 3.0, 4.0])
    temperatures = torch.tensor([0.5, 1.5, 2.5, 3.5])
    top_k = torch.tensor([60, 70, 80, 200])
    top_p = torch.tensor([0.001, 0.7, 0.9, 1.0])

    output = process_logits(logits.clone(), unique_token_ids, unique_token_counts, unique_token_lens, frequency_penalties, presence_penalties, repetition_penalties, temperatures, top_k, top_p)

    for i in range(batch_size):
        for j in range(max_seq_len): # todo
            token_id = unique_token_ids[i][j].item()
            token_count = unique_token_counts[i][j].item()
            # 1. frequency and presence penalties
            logits[i][token_id] -= frequency_penalties[i] * token_count
            if token_count > 0:
                logits[i][token_id] -= presence_penalties[i]

            # 2. repetition penalties
            if logits[i][token_id] < 0:
                logits[i][token_id] = logits[i][token_id] * repetition_penalties[i]
            else:
                logits[i][token_id] = logits[i][token_id] / repetition_penalties[i]

        # 3. temperatures 
        for j in range(vocab_size):
            logits[i][j] /= temperatures[i]

        # 4. topk
        k = min(top_k[i], vocab_size)
        top_k_values, top_k_indices = logits[i].topk(k=k, dim=-1, largest=True, sorted=True)
        logits[i].fill_(value=float('-inf'))
        for j in range(top_k_indices.numel()):
            logits[i][top_k_indices[j]] = top_k_values[j]

        # 5. topp
        logits_sorted, indices_sorted = logits[i].sort(dim=-1, descending=True)
        probs_sorted = torch.softmax(logits_sorted, dim=-1)
        probs_sorted_sum = probs_sorted.cumsum(dim=-1)
        num_masked = (probs_sorted_sum - probs_sorted > top_p[i]).sum().item()
        k = vocab_size - num_masked
        top_p_values, top_p_indices = logits_sorted[:k], indices_sorted[:k]
        logits[i].fill_(value=float('-inf'))
        for j in range(top_p_indices.numel()):
            logits[i][top_p_indices[j]] = top_p_values[j]

    output_ref = logits

    assert torch.allclose(output, output_ref, atol=1e-3, rtol=1e-3), 'wrong'