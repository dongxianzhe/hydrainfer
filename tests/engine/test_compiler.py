from hydrainfer.engine.isa import Compiler

if __name__ == '__main__':
    model_name: str = 'llava-hf/llava-1.5-7b-hf'
    from hydrainfer.model.downloader import download_hf_model
    model_path = download_hf_model(model_name)
    from transformers import AutoProcessor
    image_token_id = 32000
    processor = AutoProcessor.from_pretrained(model_path)
    prompt = f"USER: <image>\nWhat is the content of this image?\nASSISTANT:"
    token_ids = processor(
        text=prompt, 
        images=None
    )['input_ids'][0]
    from PIL import Image
    pixel_values = processor(
        text="", 
        images= Image.open('/home/xzd/projects/hydrainfer/benchmark/dataset/cherry_blossom.jpg'), 
        return_tensors="pt"
    )['pixel_values']


    def _insert_image_token_ids(token_ids: list[int], image_token_id: int, num_token_insert: int):
        # we insert 575 image_token_id before each image_token_id
        inserted_token_ids = []
        for token_id in token_ids:
            if token_id == image_token_id:
                inserted_token_ids.extend([image_token_id] * num_token_insert)
            inserted_token_ids.append(token_id)
        
        return inserted_token_ids
    token_ids = _insert_image_token_ids(token_ids, image_token_id, num_token_insert=575)

    print(f'token_ids {token_ids}')
    print(f'len(token_ids) {len(token_ids)}')

    compiler = Compiler(image_token_id=image_token_id)
    instructions = compiler.compile(token_ids, pixel_values=pixel_values)
    print(len(instructions))
    print(instructions)