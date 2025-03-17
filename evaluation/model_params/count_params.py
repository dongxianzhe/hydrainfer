"""
python count_params.py --model llava-hf/llava-1.5-7b-hf
"""
import argparse
from dxz.utils.model_utils import count_parameters
from transformers import LlavaForConditionalGeneration

def main(args: argparse.Namespace):
    print(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(args.model)
    vision_model_params = count_parameters(model.vision_tower)[0] + count_parameters(model.multi_modal_projector)[0]
    language_model_params = count_parameters(model.language_model)[0]
    print(f"vision_model_params  :{vision_model_params}")
    print(f"language_model_params:{language_model_params}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="count a model parameters")
    parser.add_argument("--model", type=str, help="model path") 
    args = parser.parse_args()
    main(args)