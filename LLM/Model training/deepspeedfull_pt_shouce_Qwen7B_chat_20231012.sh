deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed deepspeed.json \
    --stage pt \
    --model_name_or_path /bigData/models/Qwen-7B-Chat/ \
    --do_train \
    --dataset shouce \
    --finetuning_type full \
    --lora_target c_attn \
    --output_dir /bigData/trainedModel/20231012Qwen7Bchatfull/ \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --save_steps 480 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --plot_loss True\
    --fp16
