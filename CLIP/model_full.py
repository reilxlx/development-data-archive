import argparse
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

LORA_TYPE_MODEL = ["full", "visual", "text"] 


def model_full(args):
    model = CLIPModel.from_pretrained(args.model)
    if "train" == args.trainorval:
        processor = CLIPProcessor.from_pretrained(args.model, do_rescale=False)
    else:
        processor = CLIPProcessor.from_pretrained(args.model)
    tokenzier = CLIPTokenizer.from_pretrained(args.model)
    return model, processor, tokenzier


def set_trainable_params_full(model, args):
    for n, p in model.named_parameters():
        p.requires_grad = False
        if args.trained_part == 'full':
            if 'vision_model' in n or 'text_model' in n:
                p.requires_grad = True
        else:
            if 'vision_model' in n and args.trained_part == 'visual':
                p.requires_grad = True
            elif 'text_model' in n and args.trained_part == 'text':
                p.requires_grad = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--trained_part', type=str, default="full")
    parser.add_argument('--model', '-m', type=str, default="/home/temp/openaiclip-vit-large-patch14")
    parser.add_argument('--trainorval', type=str, default="train")
    args = parser.parse_args()

    model, preprocess, tokenzier = model_full(args)
    set_trainable_params_full(model,args)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)
