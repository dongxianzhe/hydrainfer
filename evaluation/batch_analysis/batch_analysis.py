from hydrainfer.model import BatchSizeProfiler, ModelFactoryConfig, ModelFactoryContext

if __name__ == '__main__':
    profiler = BatchSizeProfiler(ModelFactoryConfig(path="/models/llava-1.5-7b-hf"), ModelFactoryContext())
    profiler.profile(result_path='batchsize_analysis.json')
    profiler.plot(result_path='batchsize_analysis.json', fig_path='batch_size_analysis.pdf')