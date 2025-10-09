from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import re
import os
import json
import numpy as np
from collections import defaultdict
from metric import MethodResults, OnlineRequestOutput
from dataclasses import dataclass
from utils import load_json
from hydrainfer.utils.logger import getLogger
from typing import Optional, TypeVar, Type, Callable, Union
logger = getLogger(__file__)


@dataclass
class SLOAttainment:
    method_results: MethodResults
    ttft_attainments: list[float]
    tpot_attainments: list[float]
    slo_attainments: list[float]
    goodput: float


@dataclass
class MethodsComparionData:
    request_rate: list[float]
    ttft_slo_setting: Union[float, dict[str, float]]
    tpot_slo_setting: Union[float, dict[str, float]]
    methods_attainemnts: list[SLOAttainment]


class NameIndexer:
    def __init__(self, names: Optional[list[int]]=None):
        self.names: set[str] = set()
        self.name_to_id: dict[str, int] = {}
        self.id_to_name: list[str] = []
        if names is not None:
            for name in names:
                self.add(name)

    def add(self, name: str):
        if name in self.names:
            return
        self.names.add(name)
        self.id_to_name.append(name)
        self.name_to_id[name] = len(self.id_to_name) - 1

    def get_id(self, name: str) -> int:
        return self.name_to_id[name]

    def get_name(self, id: int) -> str:
        return self.id_to_name[id]

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, id: int) -> str:
        return self.id_to_name[id]

    def __str__(self) -> str:
        return str(self.id_to_name)

T = TypeVar('T')
def bucket_by_attr(items: list[T], key_func: Callable[[T], str]) -> dict[str, list[T]]:
    """
    Group a list of items into buckets based on a key extracted from each item.

    :param items: The list of items to be grouped.
    :param key_func: A function that extracts a string key from each item.
    :return: A dictionary where each key maps to a list of items sharing that key.
    """
    buckets = defaultdict(list)
    for item in items:
        key = key_func(item)
        buckets[key].append(item)
    return dict(buckets)


class PerformanceAnalyzer:
    """ this class is figure and table builder 
        analyzer = PerformanceAnalyzer()
        analyzer.scan_results_folder("/data1/home/dongxianzhe1/projects/hydrainfer/evaluation/slo_attainment/result")
        analyzer.parse()
        analyzer.analyze_results()
        analyzer.plot()
    """
    def __init__(self):
        self.methods_results: list[MethodResults] = []

        self.models = NameIndexer()
        self.datasets = NameIndexer()
        self.methods = NameIndexer()

    def scan_results_folder(self, result_foler_path: str) -> MethodResults:
        """ scan all the result in the result foler recursive and append the results to methods_results"""
        date_pattern = re.compile(r'^\d{8}_\d{6}$')
        for folder_name in os.listdir(result_foler_path):
            folder_path = os.path.join(result_foler_path, folder_name)
            if os.path.isdir(folder_path) and date_pattern.match(folder_name):
                logger.info(f'scanning result folder {folder_path}')
                # Iterate through all JSON files in the folder
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(folder_path, file_name)
                        try:
                            method_results = load_json(data_class=MethodResults, data_path=file_path)
                            self.methods_results.append(method_results)
                            logger.info(f'load result {file_path}')
                        except Exception as e:
                            logger.info(f"error load result {file_path}: {e}")

    def parse(self):
        """ build models, datasets, methods indexes """
        assert len(self.methods_results) > 0, 'no result is scanned'
        for method_result in self.methods_results:
            self.models.add(method_result.model)
            self.datasets.add(json.dumps(method_result.datasets))
            self.methods.add(method_result.method_name)
        logger.info(self.models)
        logger.info(self.datasets)
        logger.info(self.methods)
        
        assert all(len(method_result.results) == len(self.methods_results[0].results) for method_result in self.methods_results), f'method result is not equal'
        self.request_rates = [result.request_rate for result in self.methods_results[0].results]

    def compute_slo_attainenments(self, method_results: MethodResults, dataset_to_ttft_slo_settings: dict[str, float], dataset_to_tpot_slo_settings: dict[str, float]) -> SLOAttainment:
        def get_success_outputs(outputs: list[OnlineRequestOutput]) -> list[OnlineRequestOutput]:
            return [output for output in outputs if output.success]
        ttft_attainments = [sum(output.ttft < dataset_to_ttft_slo_settings[output.entry.dataset] for output in get_success_outputs(result.outputs)) / len(result.outputs) for result in method_results.results]
        tpot_attainments = [sum(output.tpot_statistics.p90 < dataset_to_tpot_slo_settings[output.entry.dataset] for output in get_success_outputs(result.outputs)) / len(result.outputs) for result in method_results.results]
        slo_attainments = [sum(output.ttft < dataset_to_ttft_slo_settings[output.entry.dataset] and output.tpot_statistics.p90 < dataset_to_tpot_slo_settings[output.entry.dataset] for output in get_success_outputs(result.outputs)) / len(result.outputs) for result in method_results.results]
        _, ttft_attainments = self.smooth_curve(self.request_rates, ttft_attainments)
        _, tpot_attainments = self.smooth_curve(self.request_rates, tpot_attainments)
        _, slo_attainments = self.smooth_curve(self.request_rates, slo_attainments)

        goodput = self.compute_goodput(x=self.request_rates, y=slo_attainments, goodput_threshold=0.9)
        return SLOAttainment(
            ttft_attainments=ttft_attainments, 
            tpot_attainments=tpot_attainments, 
            slo_attainments=slo_attainments, 
            method_results=method_results, 
            goodput=goodput, 
        )

    def select_best_method(self, methods_results: list[MethodResults], dataset_to_ttft_slo_settings: dict[str, float], dataset_to_tpot_slo_settings: dict[str, float]) -> SLOAttainment:
        attainments: list[SLOAttainment] = [self.compute_slo_attainenments(method_results, dataset_to_ttft_slo_settings=dataset_to_ttft_slo_settings, dataset_to_tpot_slo_settings=dataset_to_tpot_slo_settings) for method_results in methods_results]
        return max(attainments, key=lambda attainment: attainment.goodput)

    def get_intersection_points(self, x: list[float], y: list[float], target_y: float) -> list[tuple[float, float]]:
        cross_points: list[tuple[float, float]] = []
        # Check if y_i is close to target_y
        for xi, yi in zip(x, y):
            if np.isclose(yi, target_y):
                cross_points.append((xi, yi))
        # Check if each line crosses the y=target_y line
        for i in range(len(x) - 1):
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            if (y0 - target_y) * (y1 - target_y) < 0:
                t = (target_y - y0) / (y1 - y0)
                x_inter = x0 + t * (x1 - x0)
                y_inter = y0 + t * (y1 - y0)
                cross_points.append((x_inter, y_inter))
        return cross_points

    def smooth_curve(self, x: list[float], y: list[float], window_size: int=2):
        y_smooth = []
        for i in range(len(y)):
            start = max(0, i - window_size)
            end = min(i + window_size + 1, len(y))
            window = y[start:end]
            avg = sum(window) / len(window)
            y_smooth.append(avg)
        return x, y_smooth

    def compute_goodput(self, x: list[float], y: list[float], goodput_threshold: float):
        cross_points: list[tuple[float, float]] = self.get_intersection_points(x, y, goodput_threshold)
        goodput = 0
        if not cross_points:
            if y[0] < goodput_threshold:
                goodput = x[0]
            elif y[-1] > goodput_threshold:
                goodput = x[-1]
        else:
            goodput = cross_points[-1][0]
        return goodput

    def draw_slo_vertical_line(self, x_values: list[float], y_values: list[float], goodput_threshold: float, ax, **kwargs):
        cross_points: list[tuple[float, float]] = self.get_intersection_points(x_values, y_values, goodput_threshold)
        goodput = 0
        if not cross_points:
            if y_values[0] < goodput_threshold:
                ax.plot([x_values[0], x_values[0]], [-5, y_values[0]], **kwargs)
                goodput = x_values[0]
            elif y_values[-1] > goodput_threshold:
                ax.plot([x_values[-1], x_values[-1]], [-5, y_values[-1]], **kwargs)
                goodput = x_values[-1]
        for i, (x, y) in enumerate(cross_points):
            if i == len(cross_points) - 1:
                ax.plot([x, x], [0, y], **kwargs)
                goodput = x
        return goodput

    def sample_and_round_range(self, values: list[float], num_samples: int, round_to: float) -> list[float]:
        factor = 1 / round_to
        samples = np.linspace(min(values), max(values), num_samples)
        rounded = np.round(samples * factor) / factor
        return np.unique(rounded).tolist()

    def auto_select_slo_settings(self, methods_results: list[MethodResults], ttft_slo_settings: Optional[list[float]]=None, tpot_slo_settings: Optional[list[float]]=None, num_groups_selected: int=1) -> list[MethodsComparionData]:
        sample_request_rate = len(self.request_rates) // 3 * 2
        ttft_slo_settings = self.sample_and_round_range(
            values=[method_results.results[sample_request_rate].ttft_statistics.mean for method_results in methods_results], 
            num_samples=10, 
            round_to=0.2) if ttft_slo_settings is None else ttft_slo_settings
        tpot_slo_settings = self.sample_and_round_range(
            values=[method_results.results[sample_request_rate].tpot_statistics.p90 for method_results in methods_results], 
            num_samples=10, 
            round_to=0.04) if tpot_slo_settings is None else tpot_slo_settings

        logger.info(f'auto select ttft_slo_setting from {ttft_slo_settings} tpot_slo_setting from {tpot_slo_settings}')
        groups: list[MethodsComparionData] = [MethodsComparionData(
            request_rate=self.request_rates, 
            ttft_slo_setting=ttft_slo_setting, 
            tpot_slo_setting=tpot_slo_setting, 
            methods_attainemnts=[
                self.select_best_method(
                    methods_results, 
                    dataset_to_ttft_slo_settings=defaultdict(lambda : ttft_slo_setting), 
                    dataset_to_tpot_slo_settings=defaultdict(lambda : tpot_slo_setting), 
                ) for method_name, methods_results in bucket_by_attr(methods_results, lambda method_results:method_results.method_name).items()]
        ) for ttft_slo_setting in ttft_slo_settings for tpot_slo_setting in tpot_slo_settings]
        
        sorted_groups = sorted(groups, key=lambda methods_comprion_data: np.var([attainment.goodput for attainment in methods_comprion_data.methods_attainemnts]), reverse=True)
        return sorted_groups[:num_groups_selected]

    def analyze_results(self):
        self.all_methods_comparion_data: list[MethodsComparionData] = []
        for model, model_methods_results in bucket_by_attr(self.methods_results, lambda m: m.model).items():
            for dataset, model_dataset_methods_results in bucket_by_attr(model_methods_results, lambda m: json.dumps(m.datasets)).items():
                methods_comparion_data: MethodsComparionData = self.auto_select_slo_settings(model_dataset_methods_results)[0]
                self.all_methods_comparion_data.append(methods_comparion_data)
                methods_goodput = {attainment.method_results.method_name: attainment.goodput for attainment in methods_comparion_data.methods_attainemnts}
                logger.info(f'{model}, {dataset}, ttft settting {methods_comparion_data.ttft_slo_setting} tpot setting {methods_comparion_data.tpot_slo_setting} goodput {methods_goodput}')

    def plot(
        self, 
        figure_path="slo_attainment.pdf", 
        metrics: list[str]=[
            'ttft_attainments', 
            'tpot_attainments', 
            'slo_attainments', 
        ], 
        model_labels: dict[str, str]={
            "llava-hf/llava-1.5-7b-hf": "LLaVA-1.5-7B", 
            "llava-hf/llava-v1.6-vicuna-7b-hf": "LLaVA-NeXT-7B", 
            "Qwen/Qwen2-VL-7B": "Qwen2-VL-7B", 
        }, 
        dataset_labels: dict[str, str] = {
            '{"textcaps": 1, "pope": 0, "mme": 0, "text_vqa": 0, "vizwiz_vqa": 0}': "TextCaps", 
            '{"textcaps": 0, "pope": 1, "mme": 0, "text_vqa": 0, "vizwiz_vqa": 0}': "POPE", 
            '{"textcaps": 0, "pope": 0, "mme": 1, "text_vqa": 0, "vizwiz_vqa": 0}': "MME", 
            '{"textcaps": 0, "pope": 0, "mme": 0, "text_vqa": 1, "vizwiz_vqa": 0}': "TextVQA", 
            '{"textcaps": 0, "pope": 0, "mme": 0, "text_vqa": 0, "vizwiz_vqa": 1}': "VizWiz", 
        }, 
        metric_labels: dict[str, str] = {
            'ttft_attainments': 'TTFT SLO Attainment', 
            'tpot_attainments': 'TPOT SLO Attainment', 
            'slo_attainments': 'SLO Attainment', 
        }, 
        method_labels: dict[str, str] = {
            "vllm": "vLLM", 
            "sglang": "SGLang", 
            "tgi": "TGI", 
        }, 
        marker_list: list[str]=['o', 's', '^', 'v', '>', '<', 'd', 'p', '*', 'h', 'H', 'x', '+', '.', ',', '|', '_'], 
        color_list: list[str]=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"], 
        fontsize: int=22, 
    ):
        self.metrics = NameIndexer(metrics)
        n_models, n_metrics, n_datasets, n_methods = len(self.models), len(self.metrics), len(self.datasets), len(self.methods)
        n_rows, n_cols = n_models * n_metrics, len(self.datasets)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(n_rows * n_cols)
        def compute_ax_id(model_id: int, dataset_id: int, metric_id: int) -> int:
            axes_stride = (n_metrics * n_datasets, n_datasets, 1)
            ax_id = model_id * axes_stride[0] + metric_id * axes_stride[1] + dataset_id * axes_stride[2]
            return ax_id

        for methods_comparion_data in self.all_methods_comparion_data:
            for method_attainment in methods_comparion_data.methods_attainemnts:
                model_id = self.models.get_id(method_attainment.method_results.model)
                dataset_id = self.datasets.get_id(json.dumps(method_attainment.method_results.datasets))
                method_id = self.methods.get_id(method_attainment.method_results.method_name)
                for metric_id, metric in enumerate(self.metrics):
                    x = self.request_rates
                    y = getattr(method_attainment, metric)
                    ax = axes[compute_ax_id(model_id=model_id, dataset_id=dataset_id, metric_id=metric_id)]
                    ax.plot(x, y, color=color_list[method_id], marker=marker_list[method_id])
                    self.draw_slo_vertical_line(x, y, goodput_threshold=0.9, ax=ax, color=color_list[method_id], linestyle='--', alpha=0.75)
            
        for i in range(n_models):
            for j in range(n_datasets):
                for k in range(n_metrics):
                    ax = axes[compute_ax_id(i, j, k)]
                    ax.grid(False)
                    ax.set_ylim(0, 1.05)
                    formatter = FuncFormatter(lambda val, pos: f'{val * 100:.0f}')
                    ax.yaxis.set_major_formatter(formatter)
                    ax.axhline(y=0.9, color="gray", linestyle="--")

                    metric = self.metrics[k]
                    if j == 0:
                        ax.set_ylabel(metric_labels.get(self.metrics[k], self.metrics[k]), fontsize=fontsize)
                        ax.text(-0.30, 0.5, model_labels.get(self.models[i], self.models[i]), transform=ax.transAxes, ha='right', va='center', rotation=90, fontsize=fontsize)
                    if i == n_models - 1 and k == n_metrics - 1:
                        ax.set_xlabel('Request Rate (req/s)', fontsize=fontsize)
                        ax.text(7, -0.45, dataset_labels.get(self.datasets[j], self.datasets[j]), ha='center', va='bottom', fontsize=fontsize)
                    for label in ax.get_xticklabels():
                        label.set_fontsize(fontsize - 5)
                    for label in ax.get_yticklabels():
                        label.set_fontsize(fontsize - 5)
                    ax.tick_params(axis='x', which='major', length=2, width=1, direction='out', grid_color='black', grid_alpha=1)
                    ax.tick_params(axis='y', which='major', length=2, width=1, direction='out', grid_color='black', grid_alpha=1)
                    ax.tick_params(which='both', bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, direction='out')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        
        legend_labels = [method for method in self.methods]
        legend_lines = [Line2D([0], [0], color=color_list[i], marker=marker_list[i]) for i, method in enumerate(self.methods)]
        legend_n_cols = n_methods
        fig.legend(
            legend_lines, 
            legend_labels, 
            loc='upper center', ncol=legend_n_cols, fontsize=fontsize + 2, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.savefig(figure_path, bbox_inches="tight")