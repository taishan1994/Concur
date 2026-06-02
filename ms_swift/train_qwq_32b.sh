# Patch swift for datasets >= 4.0 compatibility (Json feature removed)
SWIFT_CORE=$(python3 -c "import swift.dataset.preprocessor; import os; print(os.path.dirname(swift.dataset.preprocessor.__file__))")/core.py
cp /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/swift_core_patched.py "$SWIFT_CORE"

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /ms/FM/checkpoints/Qwen-Zoo/QwQ-32B \
    --dataset /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/dataset_sft_from_arrow.parquet \
    --tuner_type lora \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --max_length 32768 \
    --sequence_parallel_size 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --optim adamw_torch \
    --output_dir ./output/qwen_qwq  \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --fp16 false \
    --bf16 true \
    --save_steps 100000 \
    --eval_steps 100000 \
    --logging_steps 10
