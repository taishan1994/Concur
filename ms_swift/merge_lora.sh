swift export \
    --model /ms/FM/checkpoints/Qwen-Zoo/QwQ-32B \
    --adapters /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/output/qwen_qwq/v2-20260523-003932/checkpoint-909 \
    --merge_lora true \
    --output_dir /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/checkpoints/KernelCoder-QwQ-32B \
    --safe_serialization true

swift export \
    --model /ms/FM/checkpoints/Qwen-Zoo/Qwen3-32B \
    --adapters /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/output/qwen3_32b/v1-20260522-144644/checkpoint-909 \
    --merge_lora true \
    --output_dir /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/checkpoints/KernelCoder-Qwen3-32B \
    --safe_serialization true

swift export \
    --model /ms/FM/checkpoints/Qwen-Zoo/Qwen3-14B \
    --adapters /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/output/qwen3_14b/v11-20260522-102327/checkpoint-909 \
    --merge_lora true \
    --output_dir /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/checkpoints/KernelCoder-Qwen3-14B \
    --safe_serialization true

swift export \
    --model /ms/FM/checkpoints/Qwen-Zoo/QwQ-32B \
    --adapters /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/output/qwen_qwq_ours/v1-20260528-021748/checkpoint-940 \
    --merge_lora true \
    --output_dir /ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/models/Qwen-QWQ-ours \
    --safe_serialization true
