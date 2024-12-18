#include "engine.h"
#include "request.h"
#include "sequence.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace mllm{
namespace py = pybind11;
using namespace pybind11::literals;

void init_output(py::module& m){
    py::class_<RequestOutput>(m, "RequestOutput")
        .def(py::init())
        .def_readwrite("output_text", &RequestOutput::output_text)
        .def_readwrite("output_token_ids", &RequestOutput::output_token_ids)
        .def_readwrite("arrival_time", &RequestOutput::arrival_time)
        .def_readwrite("first_schedule_time", &RequestOutput::first_schedule_time)
        .def_readwrite("tokens_times", &RequestOutput::tokens_times)
        .def_readwrite("finished_time", &RequestOutput::finished_time)
        .def("__repr__", [](const RequestOutput& self) {
            return "RequestOutput({})"_s.format(self.output_text);
        });
}

void init_sampling_params(py::module& m){
    py::class_<SamplingParams>(m, "SamplingParams")
        .def(py::init())
        .def_readwrite("max_tokens", &SamplingParams::max_tokens)
        .def("__repr__", [](const SamplingParams& self) {
            return "SamplingParams(max_tokens={})"_s.format(self.max_tokens);
        });
}

void init_engine(py::module& m){
    py::enum_<MemoryConfig::MemoryManagementPolicy>(m, "MemoryManagementPolicy")
        .value("VANILLA", MemoryConfig::MemoryManagementPolicy::VANILLA)
        .value("SHARED", MemoryConfig::MemoryManagementPolicy::SHARED)
        .export_values();

    py::class_<MemoryConfig>(m, "MemoryConfig")
        .def(py::init<>())
        .def_readwrite("memory_management_policy", &MemoryConfig::memory_management_policy)
        .def_readwrite("num_blocks", &MemoryConfig::num_blocks)
        .def_readwrite("block_size", &MemoryConfig::block_size);

    py::enum_<StageConfig::KVcacheEvictionPolicy>(m, "KVcacheEvictionPolicy")
        .value("NONE", StageConfig::KVcacheEvictionPolicy::NONE)
        .value("RANDOM", StageConfig::KVcacheEvictionPolicy::RANDOM)
        .value("STREAMINGLLM", StageConfig::KVcacheEvictionPolicy::STREAMINGLLM)
        .export_values();

    py::enum_<StageConfig::TokenPruningPolicy>(m, "TokenPruningPolicy")
        .value("NONE", StageConfig::TokenPruningPolicy::NONE)
        .value("RANDOM", StageConfig::TokenPruningPolicy::RANDOM)
        .value("STREAMINGLLM", StageConfig::TokenPruningPolicy::STREAMINGLLM)
        .export_values();

    py::class_<StageConfig>(m, "StageConfig")
        .def(py::init<>())
        .def_readwrite("default_max_tokens", &StageConfig::default_max_tokens)
        .def_readwrite("disaggregate_embed_prefill", &StageConfig::disaggregate_embed_prefill)
        .def_readwrite("kv_cache_eviction_policy", &StageConfig::kv_cache_eviction_policy)
        .def_readwrite("window_size", &StageConfig::window_size)
        .def_readwrite("attention_sink_size", &StageConfig::attention_sink_size)
        .def_readwrite("token_pruning_policy", &StageConfig::token_pruning_policy)
        .def_readwrite("n_embed_output_tokens", &StageConfig::n_embed_output_tokens)
        .def_readwrite("debug_mode", &StageConfig::debug_mode);

    py::enum_<SchedulerConfig::BatchPolicy>(m, "BatchPolicy")
        .value("NOBATCH", SchedulerConfig::BatchPolicy::NOBATCH)
        .value("REQUESTLEVEL", SchedulerConfig::BatchPolicy::REQUESTLEVEL)
        .value("CONTINUOUSBATCH", SchedulerConfig::BatchPolicy::CONTINUOUSBATCH)
        .export_values();

    py::enum_<SchedulerConfig::BatchPriority>(m, "BatchPriority")
        .value("PREFILL", SchedulerConfig::BatchPriority::PREFILL)
        .value("DECODE", SchedulerConfig::BatchPriority::DECODE)
        .export_values();

    py::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<>())
        .def_readwrite("batch_policy", &SchedulerConfig::batch_policy)
        .def_readwrite("priority", &SchedulerConfig::priority)
        .def_readwrite("max_running_sequences", &SchedulerConfig::max_running_sequences)
        .def_readwrite("max_batch_fill_tokens", &SchedulerConfig::max_batch_fill_tokens)
        .def_readwrite("max_batch_embed_images", &SchedulerConfig::max_batch_embed_images)
        .def_readwrite("batch_embed_fill", &SchedulerConfig::batch_embed_fill)
        .def_readwrite("debug_mode", &SchedulerConfig::debug_mode);

    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("num_handling_threads", &EngineConfig::num_handling_threads)
        .def_readwrite("model_path", &EngineConfig::model_path)
        .def_readwrite("memory_config", &EngineConfig::memory_config)
        .def_readwrite("stage_config", &EngineConfig::stage_config)
        .def_readwrite("scheduler_config", &EngineConfig::scheduler_config)
        .def_readwrite("batch_image_embed_forward", &EngineConfig::batch_image_embed_forward);

    py::class_<std::future<bool>>(m, "Future")
        .def("wait",
            &std::future<bool>::wait,
            py::call_guard<py::gil_scoped_release>())
        .def("get",
            &std::future<bool>::get,
            py::call_guard<py::gil_scoped_release>());

    py::class_<BatchFuture>(m, "BatchFuture")
        .def("wait", &BatchFuture::wait, py::call_guard<py::gil_scoped_release>())
        .def("get", &BatchFuture::get, py::call_guard<py::gil_scoped_release>());

    py::class_<Engine>(m, "FastEngine")
        .def(py::init<const EngineConfig&>(), py::arg("config"))
        .def("step", &Engine::step, py::call_guard<py::gil_scoped_release>())
        .def("run_until_complete", &Engine::run_until_complete, py::call_guard<py::gil_scoped_release>())
        .def("add_request_async", &Engine::add_request_async, py::call_guard<py::gil_scoped_release>())
        .def("add_requests_async", &Engine::add_requests_async, py::call_guard<py::gil_scoped_release>())
        .def("stop", &Engine::stop, py::call_guard<py::gil_scoped_release>());

}

PYBIND11_MODULE(PY_MODULE_NAME, m) {
    m.doc() = "fast engine implemented with C++";
    // glog and glfag will be initialized in folly::init
    //   int argc = 0;
    //   char** argv = nullptr;
    //   folly::Init init(&argc, &argv);
    //   m.def("get_metrics", &get_metrics);
    
    init_sampling_params(m);
    init_output(m);
    init_engine(m);
}


}