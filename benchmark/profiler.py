import numpy as np
from hydrainfer.model import BatchSizeProfiler, ModelFactoryConfig, ModelFactoryContext, getModelFactory

from hydrainfer.utils.json_utils import load_json
from data_preprocess import PreProcessedResult

def partition_nodes(n_nodes: int, ratios: list[float]) -> list[int]:
    num_classes = len(ratios)
    if n_nodes < num_classes:
        raise Exception(f'n_nodes {n_nodes} can not be partitioned with ratios {ratios}')

    ratios = np.array(ratios, dtype=float)
    total_ratio = np.sum(ratios)

    # 1. first allocate each class one node
    base_alloc = np.ones(num_classes, dtype=int)
    remaining = n_nodes - num_classes

    # 2. if ratios is all zero divide zero
    if total_ratio == 0:
        for i in range(remaining):
            base_alloc[i % num_classes] += 1
        return base_alloc.tolist()

    # 3.distribute the remaining part proportionally.
    norm_ratios = ratios / total_ratio
    raw_alloc = norm_ratios * remaining
    int_alloc = np.floor(raw_alloc).astype(int)
    leftover = remaining - np.sum(int_alloc)

    # 4.distribute the remainder, prioritize those with the largest rounding errors.
    residuals = raw_alloc - int_alloc
    for i in np.argsort(-residuals):
        if leftover == 0:
            break
        int_alloc[i] += 1
        leftover -= 1

    final_alloc = base_alloc + int_alloc
    return final_alloc.tolist()


def estimate_converged_value(x: list[float], y: list[float], window_size: int=5, rel_threshold: float=0.01):
    """
    Detects the convergence region based on a relative threshold and estimates the converged value.

    Parameters:
    - x: List of independent variable values (optional, used to preserve order)
    - y: List of dependent variable values (assumed to first increase then stabilize)
    - window_size: Size of the sliding window used to detect the stable region
    - rel_threshold: Relative change threshold (as a fraction of the y-value range)

    Returns:
    - Converged value (float), or None if no stable region is detected
    """
    y = np.array(y)
    if len(y) < window_size + 2:
        return None

    y_range = np.max(y)
    if y_range == 0:
        return y[0]

    abs_threshold = rel_threshold * y_range
    diffs = np.abs(np.diff(y))

    for i in range(len(diffs) - window_size, 0, -1):
        window = diffs[i:i+window_size]
        if np.all(window < abs_threshold):
            return float(np.mean(y[i:i+window_size+1]))

    return None


class DisaggregationProfiler:
    def __init__(self) -> None:
        pass


if __name__ == '__main__':
    # config, context = ModelFactoryConfig(path="/models/llava-1.5-7b-hf"), ModelFactoryContext()
    config, context = ModelFactoryConfig(path="/models/llava-v1.6-vicuna-7b-hf"), ModelFactoryContext()
    result = load_json(PreProcessedResult, './cache/lmms-lab-TextCaps_test.log')

    model_factory = getModelFactory(config, context)
    image_token_caculator = model_factory.getVisionModelConfig().image_token_caculator

    total_encode_tokens: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    for request in result.requests:
        n_image_tokens = sum([image_token_caculator.get_num_image_tokens(image_size) for image_size in request.images_size])
        n_input_tokens = request.n_input_tokens
        n_output_tokens = request.n_output_tokens

        total_encode_tokens += n_image_tokens
        total_prefill_tokens += n_input_tokens
        total_decode_tokens += n_output_tokens

    # profiler = BatchSizeProfiler(config, context)
    # profiler.profile_prefill(batch_size=4096, n_prompt_tokens)
    from hydrainfer.model import BatchSizeAnalysisResult
    batch_analysis_results = load_json(BatchSizeAnalysisResult, '/data1/home/dongxianzhe1/projects/hydrainfer/evaluation/batch_analysis/batchsize_analysis.json')

    t1 = total_encode_tokens // batch_analysis_results.encode[1].throughput
    t2 = total_prefill_tokens // batch_analysis_results.prefill[1].throughput
    t3 = total_decode_tokens // batch_analysis_results.decode[3].throughput
    print(t1, t2, t3)

    s = t1 + t2 + t3

    n_nodes = 8

    partitions = partition_nodes(n_nodes=n_nodes, ratios=[t1, t2, t3]) 
    print(f'partitions {partitions}')