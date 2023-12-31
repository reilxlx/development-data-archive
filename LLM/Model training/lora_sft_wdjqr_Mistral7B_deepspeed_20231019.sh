deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed deepspeed.json \
    --stage sft \
    --model_name_or_path /bigData/models/Mistral-7B-v0.1/ \
    --do_train \
    --dataset wdjqr \
    --template default \
    --finetuning_type lora \
    --lora_target all \
    --output_dir  /bigData/trainedModel/20231019Mistral7B/ \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1000 \
    --save_steps 141384 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
