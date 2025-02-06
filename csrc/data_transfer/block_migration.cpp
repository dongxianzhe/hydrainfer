#include "block_migration.h"
#include<torch/torch.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>
#include<vector>

namespace mllm::migration{

#define CUDA_CHECK(cmd) do { \
    cudaError_t result = cmd; \
    if (result != cudaSuccess) { \
        printf("[ERROR] CUDA error %s:%d '%s': (%d) %s\n", __FILE__, __LINE__, #cmd, (int)result, cudaGetErrorString(result)); \
        exit(-1); \
    } \
} while(0)

// Some stuff for indexing into an 1-D array
#define INDEX_2D(dim1, dim2, index1, index2) \
    (((int64_t)index1) * (dim2) + (index2))
#define INDEX_3D(dim1, dim2, dim3, index1, index2, index3) \
    (((int64_t)index1) * (dim2) * (dim3) + ((int64_t)index2) * (dim3) + (index3))
#define INDEX_4D(dim1, dim2, dim3, dim4, index1, index2, index3, index4) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) + ((int64_t)index2) * (dim3) * (dim4) + ((int64_t)index3) * (dim4) + (index4))
#define INDEX_5D(dim1, dim2, dim3, dim4, dim5, index1, index2, index3, index4, index5) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) * (dim5) + ((int64_t)index2) * (dim3) * (dim4) * (dim5) + ((int64_t)index3) * (dim4) * (dim5) + (index4) * (dim5) + (index5))

/*
The following two functions convert cudaIpcMemHandle_t to/from bytes
We need this because we need to pass cudaIpcMemHandle_t to Python
*/

static std::vector<int64_t> cudaIpcMemHandle2Bytes(const cudaIpcMemHandle_t &handle) {
	std::vector<int64_t> result;
	for (size_t i = 0; i < sizeof(handle); ++i) {
		result.push_back(((uint8_t*) &handle)[i]);
	}
	return result;
}

static cudaIpcMemHandle_t bytes2CudaIpcMemHandle(const std::vector<int64_t> &bytes) {
    CHECK(bytes.size() == sizeof(cudaIpcMemHandle_t));
	cudaIpcMemHandle_t result;
	for (size_t i = 0; i < sizeof(result); ++i) {
		((uint8_t*) &result)[i] = bytes[i];
	}
	return result;
}

/*
get_ipc_mem_handle: Get the IPC memory handle of a tensor
The returned handle can be used to open the tensor in another process.
*/
std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor) {
	cudaIpcMemHandle_t handle;
	CUDA_CHECK(cudaIpcGetMemHandle(&handle, tensor.data_ptr()));
	return cudaIpcMemHandle2Bytes(handle);
}

static std::vector<void *> prefill_dev_ptr_vec;

/*
Register the IPC memory handle of a tensor
The returned device ptr index should be used by function migrate_blocks()
User need to decide which index should be used, since the layer_id, rank, tp size, pp size is unknown.
*/

int64_t register_ipc_mem_handle(std::vector<int64_t> kv_cache_handle_vec) {
	// Convert the handles to cudaIpcMemHandle_t
	const cudaIpcMemHandle_t kv_cache_handle = bytes2CudaIpcMemHandle(kv_cache_handle_vec);
	void* devPtr = nullptr;
	cudaError_t err = cudaIpcOpenMemHandle(&devPtr, kv_cache_handle, cudaIpcMemLazyEnablePeerAccess);
	if (err == cudaErrorPeerAccessUnsupported) {
		printf("Error: Peer-to-peer access is unsupported on this platform.\n");
		return -1;
	} 
	prefill_dev_ptr_vec.push_back(devPtr);
	return prefill_dev_ptr_vec.size() - 1;
}

/*
migrate_blocks: Migrate blocks from the context stage engine to the decoding stage engine

This function is called by every decoding stage worker when the decoding
stage engine decides to migrate some blocks from the context stage engine
to the decoding stage engine.

In the following code, "prefill_start_head" and "prefill_end_head" is related to "prefill_dev_ptr_index".
"decoding_start_head" and "decoding_end_head" is related to the called worker.

User need to make sure that "prefill_dev_ptr_index" and "decoding_worker_kv_cache" have the same layer id.

Here we do not pass a cudaStream to the function. Instead we use the current
stream indicated by at::cuda::getCurrentCUDAStream(). So it is python's
responsibility to set the current stream before calling this function.
*/

void migrate_blocks(
	const int64_t prefill_start_head,
	const int64_t prefill_end_head,
	
	// Block indexes of the prefill stage engine
	const std::vector<int64_t> &prefill_block_indexes,

	const int64_t decoding_start_head,
	const int64_t decoding_end_head,

	// Block indexes of the decoding stage engine
	const std::vector<int64_t> &decoding_block_indexes,

	const int64_t prefill_dev_ptr_index,
	const int64_t num_heads,

	// The decoding stage [worker, ve, layer]'s KV cache
	// [2, num_blocks, block_size, num_kv_heads, head_size]
	torch::Tensor decoding_worker_kv_cache
) {
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	CHECK(decoding_worker_kv_cache.is_contiguous());

	// for kv cache block migration num_cache_pool is 2 because each token has key cache and value cache
	// for image cache block migration num_cache_pool is 1
	const int64_t num_cache_pool = decoding_worker_kv_cache.size(0);
	// Calculate some misc stuff
	const int64_t num_blocks = decoding_worker_kv_cache.size(1);
	const int64_t block_size = decoding_worker_kv_cache.size(2);
	const int64_t heads_per_decoding_worker = decoding_worker_kv_cache.size(3);
	const int64_t head_size = decoding_worker_kv_cache.size(4);

	const int64_t heads_per_prefill_worker = prefill_end_head - prefill_start_head;
	const int64_t num_blocks_to_copy = decoding_block_indexes.size();
	const int64_t dtype_size = decoding_worker_kv_cache.dtype().itemsize();

	const int64_t overlap_start_head = std::max(prefill_start_head, decoding_start_head);
	const int64_t overlap_end_head = std::min(prefill_end_head, decoding_end_head);
	CHECK(overlap_start_head < overlap_end_head);

	// kv cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
	for (int64_t block_id = 0; block_id < num_blocks_to_copy; ++block_id) {
		const int64_t prefill_block_index = prefill_block_indexes[block_id];
		const int64_t decoding_block_index = decoding_block_indexes[block_id];
		for (int64_t is_value = 0; is_value < num_cache_pool; ++is_value) {
			char* prefill_worker_base_ptr = (char*)(prefill_dev_ptr_vec.at(prefill_dev_ptr_index));
			if (!prefill_worker_base_ptr) {
				fprintf(stderr, "Error: registered prefill_worker_base_ptr is null\n");
				exit(1);
			}
			const int64_t decode_bias = INDEX_5D(
						0, num_blocks, block_size, heads_per_decoding_worker, head_size,
						is_value,
						decoding_block_index,
						0,
						overlap_start_head - decoding_start_head,
						0);
			const int64_t prefill_bias = INDEX_5D(
						0, num_blocks, block_size, heads_per_prefill_worker, head_size,
						is_value,
						prefill_block_index,
						0,
						overlap_start_head - prefill_start_head,
						0);

			if (heads_per_prefill_worker == heads_per_decoding_worker) {
				CUDA_CHECK(cudaMemcpyAsync(
					(char*) (decoding_worker_kv_cache.data_ptr())
						+ decode_bias * dtype_size,
					prefill_worker_base_ptr
						+ prefill_bias * dtype_size,
					(size_t) (block_size * heads_per_prefill_worker * head_size * dtype_size),
					cudaMemcpyDeviceToDevice,
					stream	
					)
				);
			} else {
				CUDA_CHECK(cudaMemcpy2DAsync(
					(char*) (decoding_worker_kv_cache.data_ptr())
						+ decode_bias * dtype_size,
					(size_t) (heads_per_decoding_worker * head_size * dtype_size),
					prefill_worker_base_ptr
						+ prefill_bias * dtype_size,
					(size_t) (heads_per_prefill_worker * head_size * dtype_size),
					(size_t) ((overlap_end_head - overlap_start_head) * head_size * dtype_size),
					(size_t) (block_size),
					cudaMemcpyDeviceToDevice,
					stream
					)
				);
			}
		}
	}
}

}