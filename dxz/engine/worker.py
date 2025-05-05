import ray
import torch
import argparse
from torch import Tensor
from typing import Optional
from dxz.model.model_factory import LanguageModelParameters, LanguageModelOutput, VisionModelParameters, VisionModelOutput, ModelFactoryConfig, ModelFactoryContext, ModelFactory, getModelFactory
from dxz.model_parallel.process_group import init_global_process_group, ParallelConfig
from dataclasses import dataclass, field, fields


@dataclass
class WorkerConfig:
    use_ray: bool = False
    init_method: str = 'tcp://localhost:9876'
    has_vision_model: bool = True
    has_language_model: bool = True
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    model: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)


@dataclass
class WorkerContext:
    pass


class Worker:
    def execute_language_model(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        raise NotImplementedError

    def execute_vision_model(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        raise NotImplementedError


class VanillaWorker(Worker):
    def __init__(self, config: WorkerConfig, context: WorkerContext):
        model_factory_context = ModelFactoryContext(process_group=None)
        model_factory: ModelFactory = getModelFactory(config.model, model_factory_context)
        if config.has_vision_model:
            self.vision_model = model_factory.getVisionModel() 
        if config.has_language_model:
            self.language_model = model_factory.getLanguageModel() 

    @torch.inference_mode()
    def execute_language_model(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        return self.language_model.forward(input_ids, image_features, position_ids, model_params)

    @torch.inference_mode()
    def execute_vision_model(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        return self.vision_model.forward(pixel_values, model_params)


class RayWorker(Worker):
    def __init__(self, config: WorkerConfig, context: WorkerContext, rank: int):
        init_global_process_group(backend='nccl', world_size=config.parallel_config.world_size, rank=rank, init_method=config.init_method)
        from dxz.model_parallel.process_group import ProcessGroup
        model_factory_context = ModelFactoryContext(
            process_group=ProcessGroup(
                rank=rank, 
                global_ranks=list(range(config.parallel_config.world_size))
            )
        )
        model_factory: ModelFactory = getModelFactory(config.model_factory_config, model_factory_context)
        self.vision_model = model_factory.getVisionModel() 
        self.language_model = model_factory.getLanguageModel() 
        print('ray worker init finished')

    def execute_language_model(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        return self.language_model.forward(input_ids, image_features, position_ids, model_params)

    def execute_vision_model(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        return self.vision_model.forward(pixel_values, model_params)


class RayWorkers(Worker):
    def __init__(self, config: WorkerConfig, context: WorkerContext):
        super().__init__()
        assert config.parallel_config.pp_size == 1, 'now we only support pp_size=1'
        from dxz.utils.ray_utils import launch_ray_cluster, get_ip_address
        launch_ray_cluster(is_head_node=True, head_node_ip=get_ip_address(), ray_cluster_port=8765, namespace='ray_workers')
        self.workers: list[RayWorker] = []
        for rank in range(config.parallel_config.world_size):
            worker = ray.remote(num_cpus=0, num_gpus=1)(RayWorker).remote(config, context, rank)
            self.workers.append(worker)
        print('ray workers init finished')

    def execute_language_model(self, input_ids: list[Tensor], image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        print('execute language model')
        objs = []
        for worker in self.workers:
            obj = worker.execute_language_model.remote(input_ids, image_features, position_ids, model_params)
            objs.append(obj)
        results = ray.get(objs)
        print('execute language model finished')
        return results[0]

    def execute_vision_model(self, pixel_values: Tensor, model_params: VisionModelParameters) -> VisionModelOutput:
        objs = []
        for worker in self.workers:
            obj = worker.execute_vision_model.remote(pixel_values, model_params)
            objs.append(obj)
        results = ray.get(objs)
        return results[0]


def getWorker(config: WorkerConfig, context: WorkerContext) -> Worker:
    if config.use_ray:
        return RayWorkers(config, context)
    # todo the yaml config does not has property function
    # if config.parallel_config.is_parallel:
    #     return RayWorkers(config, context)
    return VanillaWorker(config, context)