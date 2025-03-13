import sys
import argparse
from tqdm import tqdm
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from mme.MMETestPaper import MMETestPaper
from textcaps.TextCapsDataset import TextCapsDataset
from pope.POPETestPaper import POPETestPaper
from vega.VEGADataset import VEGADataset

def get_dataset(name: str):
    if name == 'mme':
        return MMETestPaper(path='mme')
    elif name == 'textcaps':
        return TextCapsDataset(path='textcaps')
    elif name == 'vega':
        return VEGADataset(path='vega')
    else:
        raise Exception(f'invalid dataset {name}')

def analysis_dataset(args: argparse.Namespace):
    device = torch.device('cuda:0')
    model_name_or_path = "llava-hf/llava-1.5-7b-hf"

    # model = LlavaForConditionalGeneration.from_pretrained(
    #     pretrained_model_name_or_path=model_name_or_path, device_map=device, torch_dtype=torch.half
    # )
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    tokenizer = processor.tokenizer

    dataset = get_dataset(args.dataset)
    if isinstance(dataset, VEGADataset):
        for question in tqdm(dataset.questions):
            n_tokens_without_image = len(tokenizer.encode(f"USER:\n{question.context}\n{question.question}?\nASSISTANT:"))
            n_output_tokens = len(tokenizer.encode(question.answer))
            print(n_tokens_without_image, n_output_tokens)
        return

    for question in tqdm(dataset.questions):
        inputs = processor(
            # text=f"<image>{question.question}", 
            text=f"USER: <image>\n{question.question}?\nASSISTANT:", 
            images=question.image,
            return_tensors="pt"
        )
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=512)

        token_ids = tokenizer.encode(question.question)
        n_tokens_without_image = len(token_ids)
        n_tokens_with_image = inputs['input_ids'].numel()
        n_output_tokens = generated_ids.numel() - n_tokens_with_image

        response = processor.batch_decode(
            generated_ids[:, n_tokens_with_image:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if args.debug:
            print(f'<question> {question.question} <response> {response}')
        print(n_tokens_without_image, n_tokens_with_image, n_output_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analysis dataset workload', conflict_handler='resolve')
    parser.add_argument(
        "--dataset", 
        type=str,
        choices=["mme", "textcaps", "vega"],
        default="mme",
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug mode'
    )
    args = parser.parse_args()
    analysis_dataset(args)