import argparse
import json
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Dataset
from openai import OpenAI

# ============================================================
# Defaults (overridable via CLI)
# ============================================================
API_PARAMS = {
    "url": "http://192.168.112.27:11378/v1",
    "api_key": "none",
    "model_id": "/ms/FM/checkpoints/Qwen-Zoo/QwQ-32B",
    "temperature": 1.0,
    "max_tokens": 32768,
}

# ============================================================
# Concurrency Configuration
# ============================================================
NUM_THREADS = 8
NUM_GPUS = 8
OUTPUT_DIR = "data/kernelbench_results_qwen-qwq-test"
KB_DIR = "/ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/KernelBench/KernelBench"
KB_LEVELS = ["level1"]

# ============================================================
# Template
# ============================================================
TEMPLATE = """You are a Machine Learning Engineer trying to write custom cuda kernels to replace the pytorch operators in the given architecture to get speedups. You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom cuda kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination. For [Imports], you will likely need but not limited to the following libraries:
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```
Here's an example to show you the syntax of inline embedding custom operators from the cuda kernel in torch:
The pytorch module needed to be optimize is:
```
$ref_arch_torch
```

The example new arch with custom cuda kernels looks like this:
```
$ref_arch_kernel
```

And the PyTorch code you need to optimize is:
```
$code
```

Optimize the architecture named Model with custom cuda kernels! Optimize the architecture named Model with custom cuda kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!

"""

# ============================================================
# GPU pool (shared across threads, each thread acquires a GPU ID)
# ============================================================
gpu_pool = None  # initialized in main(): multiprocessing.Manager().Queue()


def inference(messages, openai_client, model_id, api_params):
    """Call OpenAI-format API, return response text."""
    try:
        request_kwargs = {
            "model": model_id,
            "messages": messages,
            "max_tokens": api_params["max_tokens"],
            "temperature": api_params["temperature"],
        }
        if "top_p" in api_params:
            request_kwargs["top_p"] = api_params["top_p"]

        extra_body = {}
        if "top_k" in api_params:
            extra_body["top_k"] = api_params["top_k"]
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = openai_client.chat.completions.create(**request_kwargs)

        message = response.choices[0].message
        try:
            think_content = message.reasoning_content
        except AttributeError:
            think_content = None

        if think_content:
            main_content = message.content if message.content else ""
            return f"<think>\n{think_content}\n</think>\n\n{main_content}"
        return message.content if message.content else ""

    except Exception as e:
        print(f"Inference error: {e}")
        return ""


def extract_code_from_response(response_content):
    """Extract Python code from model response.

    Handles: <think> blocks, ```python code blocks, and raw code.
    """
    text = response_content

    # 1. Strip <think>...</think> reasoning block if present
    # text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.split("</think>")[-1]

    # 2. Try ```python ... ``` first
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # 3. Try bare ``` ... ```
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # 4. Fallback: return text as-is (it's already raw code after removing think)
    return text.strip()


def _subprocess_eval_worker(q, device_id, gen_code, ref_code_str):
    """Module-level worker for spawn-based subprocess evaluation."""
    try:
        import time

        import torch
        import torch.nn as nn

        torch.cuda.set_device(device_id)
        device = f"cuda:{device_id}"

        # ---- Step 1: Load reference model ----
        ref_ns = {
            "torch": torch,
            "nn": nn,
            "F": torch.nn.functional,
            "math": __import__("math"),
        }
        exec(ref_code_str, ref_ns)
        Model = ref_ns["Model"]
        get_inputs = ref_ns.get("get_inputs", None)
        get_init_inputs = ref_ns.get("get_init_inputs", None)

        # IMPORTANT: Set seed before getting inputs (like kernelbench)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Get init args — format: get_init_inputs() returns [args_list, kwargs_dict]
        if get_init_inputs is not None:
            init_data = get_init_inputs()
        else:
            init_data = []
        if len(init_data) == 2 and isinstance(init_data[0], (list, tuple)) and isinstance(init_data[1], dict):
            init_args, init_kwargs = init_data
        elif isinstance(init_data, (list, tuple)):
            init_args, init_kwargs = init_data, {}
        else:
            init_args, init_kwargs = [init_data], {}

        # Get inputs — format: get_inputs() returns [tensor1, tensor2, ...]
        if get_inputs is not None:
            inputs = get_inputs()
        else:
            inputs = []
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # Build device and cpu copies
        device_inputs = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs
        ]
        device_init_args = [
            arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in init_args
        ]
        device_init_kwargs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in init_kwargs.items()
        }
        cpu_init_args = [
            arg.cpu().clone() if isinstance(arg, torch.Tensor) else arg for arg in init_args
        ]
        cpu_init_kwargs = {
            k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in init_kwargs.items()
        }
        cpu_inputs = [
            inp.cpu().clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs
        ]

        # Create reference model and get output
        # IMPORTANT: Set seed AGAIN before creating model (like kernelbench)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        ref_model = Model(*device_init_args, **device_init_kwargs)
        ref_model.to(device=device, dtype=torch.float32)
        ref_model.eval()

        with torch.no_grad():
            ref_output = ref_model(*device_inputs)

        if isinstance(ref_output, tuple):
            ref_outputs = list(ref_output)
        else:
            ref_outputs = [ref_output]

        # ---- Step 2: Benchmark reference model ----
        ref_model2_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in cpu_init_kwargs.items()}
        ref_model2 = Model(*[a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_init_args], **ref_model2_kwargs)
        ref_model2.to(device=device, dtype=torch.float32)
        ref_model2.eval()

        for _ in range(5):
            ref_model2(*[a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_inputs])
        torch.cuda.synchronize()

        num_iters = 50
        start = time.perf_counter()
        for _ in range(num_iters):
            ref_model2(*[a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_inputs])
        torch.cuda.synchronize()
        ref_time = (time.perf_counter() - start) / num_iters

        del ref_model2
        torch.cuda.empty_cache()

        # ---- Step 3: Load generated (optimized) model ----
        gen_ns = {
            "torch": torch,
            "nn": nn,
            "F": torch.nn.functional,
            "math": __import__("math"),
        }
        exec(gen_code, gen_ns)
        ModelNew = gen_ns["ModelNew"]

        # IMPORTANT: Set seed before creating model (like kernelbench)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        new_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in cpu_init_kwargs.items()}
        new_model = ModelNew(*[a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_init_args], **new_kwargs)
        new_model.to(device=device, dtype=torch.float32)
        new_model.eval()

        # ---- Step 4: Check correctness (with kernelbench's multi-trial approach) ----
        # Follow kernelbench's approach: generate trial seeds, set seed before each operation,
        # and re-load models to device each trial to ensure consistent weights
        num_correct_trials = 5
        pass_count = 0

        # Generate trial seeds like kernelbench
        torch.manual_seed(42)
        trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)]

        for trial in range(num_correct_trials):
            trial_seed = trial_seeds[trial]

            # Set seed before getting inputs (like kernelbench)
            torch.manual_seed(trial_seed)
            torch.cuda.manual_seed(trial_seed)
            trial_inputs = get_inputs()
            trial_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in trial_inputs]

            # Set seed before loading model to device (like kernelbench)
            # IMPORTANT: Reuse the model instance instead of recreating it
            # This matches the official kernelbench behavior
            torch.manual_seed(trial_seed)
            torch.cuda.manual_seed(trial_seed)
            ref_model_trial = ref_model.to(device=device, dtype=torch.float32)

            torch.manual_seed(trial_seed)
            torch.cuda.manual_seed(trial_seed)
            new_model_trial = new_model.to(device=device, dtype=torch.float32)
            new_model_trial.eval()

            with torch.no_grad():
                ref_output = ref_model_trial(*trial_inputs)
                new_output = new_model_trial(*trial_inputs)

            if isinstance(ref_output, tuple):
                ref_outputs_trial = list(ref_output)
            else:
                ref_outputs_trial = [ref_output]

            if isinstance(new_output, tuple):
                new_outputs_trial = list(new_output)
            else:
                new_outputs_trial = [new_output]

            trial_correct = True
            for ref_out, new_out in zip(ref_outputs_trial, new_outputs_trial):
                if ref_out.shape != new_out.shape:
                    trial_correct = False
                    break
                if not torch.allclose(ref_out, new_out, atol=1e-4, rtol=1e-4):
                    trial_correct = False
                    break

            if trial_correct:
                pass_count += 1

            # Clean up trial models
            del ref_model_trial, new_model_trial
            torch.cuda.empty_cache()

        correct = (pass_count == num_correct_trials)
        error_detail = None
        if not correct:
            with torch.no_grad():
                new_output = new_model(*device_inputs)
            if isinstance(new_output, tuple):
                new_outputs = list(new_output)
            else:
                new_outputs = [new_output]
            for idx, (ref_out, new_out) in enumerate(zip(ref_outputs, new_outputs)):
                if ref_out.shape != new_out.shape:
                    error_detail = f"Output {idx} shape mismatch"
                    break
                if not torch.allclose(ref_out, new_out, atol=1e-4, rtol=1e-4):
                    max_diff = (ref_out - new_out).abs().max().item()
                    error_detail = f"Output {idx} value mismatch: max_diff={max_diff:.6f}"
                    break

        # Done with correctness check, delete ref_model
        del ref_model
        torch.cuda.empty_cache()

        # ---- Step 5: Benchmark optimized model ----
        del new_model
        torch.cuda.empty_cache()

        new_model2_kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in cpu_init_kwargs.items()}
        new_model2 = ModelNew(*[a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_init_args], **new_model2_kwargs)
        new_model2.to(device=device, dtype=torch.float32)
        new_model2.eval()

        for _ in range(5):
            new_model2(*[a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_inputs])
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            new_model2(*[a.to(device) if isinstance(a, torch.Tensor) else a for a in cpu_inputs])
        torch.cuda.synchronize()
        new_time = (time.perf_counter() - start) / num_iters

        speedup = ref_time / new_time if new_time > 0 else 0.0

        q.put({
            "compiled": True,
            "correctness": correct,
            "speedup": round(speedup, 4),
            "ref_runtime": round(ref_time, 6),
            "new_runtime": round(new_time, 6),
            "error_message": error_detail,
        })

    except Exception as e:
        q.put({
            "compiled": False,
            "correctness": False,
            "speedup": 0.0,
            "ref_runtime": 0.0,
            "new_runtime": 0.0,
            "error_message": str(e),
        })


def evaluate_inline_kernel(generated_code, ref_code, gpu_id, timeout=600):
    """Evaluate inline CUDA kernel in a subprocess (spawn for CUDA isolation)."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()

    p = ctx.Process(
        target=_subprocess_eval_worker,
        args=(queue, gpu_id, generated_code, ref_code),
    )
    p.start()

    try:
        result = queue.get(timeout=timeout)
    except Exception as e:
        result = {
            "compiled": False,
            "correctness": False,
            "speedup": 0.0,
            "error_message": f"Subprocess timeout: {str(e)}",
        }

    p.join(timeout=5)
    if p.is_alive():
        p.terminate()

    return result


def process_one_sample(level, problem_name, pytorch_code, ref_arch_torch, ref_arch_kernel, output_dir):
    """Process a single KernelBench sample: API call + inline evaluation."""
    label = f"{level}_{problem_name}"
    output_file = os.path.join(output_dir, f"{label}.json")

    # Skip if already done
    if os.path.exists(output_file):
        print(f"[Thread] 跳过 {label}: 已存在")
        return None

    print(f"[Thread] 开始: {label}")

    # ---- Step 1: API Inference ----
    openai_client = OpenAI(base_url=API_PARAMS["url"], api_key=API_PARAMS["api_key"])

    user_content = (TEMPLATE
                    .replace("$ref_arch_torch", ref_arch_torch)
                    .replace("$ref_arch_kernel", ref_arch_kernel)
                    .replace("$code", pytorch_code))

    messages = [{"role": "user", "content": user_content}]
    assistant_content = inference(messages, openai_client, API_PARAMS["model_id"], API_PARAMS)

    if not assistant_content:
        print(f"[Thread] {label}: 推理失败")
        return None

    messages.append({"role": "assistant", "content": assistant_content})

    # ---- Step 2: Extract generated code ----
    generated_code = extract_code_from_response(assistant_content)
    if not generated_code:
        print(f"[Thread] {label}: 未能从响应中提取代码")
        generated_code = assistant_content  # fallback

    # ---- Step 3: Evaluate (with GPU pool) ----
    feedback = None
    gpu_id = gpu_pool.get()  # 从池中获取一个可用 GPU ID
    try:
        feedback = evaluate_inline_kernel(
            generated_code,
            pytorch_code,
            gpu_id=gpu_id,
            timeout=600,
        )
    finally:
        gpu_pool.put(gpu_id)  # 归还 GPU ID

    # ---- Step 4: Save results ----
    speedup = feedback.get("speedup", 0.0) if feedback else 0.0
    correct = feedback.get("correctness", False) if feedback else False

    result = {
        "level": level,
        "problem_name": problem_name,
        "messages": messages,
        "generated_code": generated_code,
        "feedback": feedback,
        "speedup": speedup,
        "correctness": correct,
        "original_python_code": pytorch_code,
        "timestamp": time.time(),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    status = f"speedup={speedup:.2f}x, correct={correct}" if feedback else "评估跳过"
    print(f"[Thread] {label}: 完成 ({status})")
    return result


def load_kernelbench_tasks():
    """Load all KernelBench problems from level1-4 directories."""
    tasks = []
    for level in KB_LEVELS:
        lvl_dir = os.path.join(KB_DIR, level)
        if not os.path.isdir(lvl_dir):
            continue
        for fname in sorted(os.listdir(lvl_dir)):
            if not fname.endswith('.py'):
                continue
            fpath = os.path.join(lvl_dir, fname)
            with open(fpath) as f:
                code = f.read()
            # Extract problem name from filename: "19_ReLU.py" → "ReLU"
            name = fname.rsplit('.', 1)[0]
            tasks.append((level, name, code, fpath))
    return tasks


def parse_args():
    p = argparse.ArgumentParser(description="KernelBench evaluation with vLLM/SGLang API")
    p.add_argument("--url", default=None, help="API base URL (default: %(default)s)")
    p.add_argument("--model", default=None, help="Model ID for API calls")
    p.add_argument("--output-dir", default=None, help="Output directory for results")
    p.add_argument("--num-threads", type=int, default=None, help="Number of parallel API threads")
    p.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs for eval (must match CUDA_VISIBLE_DEVICES count)")
    return p.parse_args()


def main():
    global gpu_pool, API_PARAMS, NUM_THREADS, NUM_GPUS, OUTPUT_DIR

    args = parse_args()
    if args.url:
        API_PARAMS = {**API_PARAMS, "url": args.url}
    if args.model:
        API_PARAMS = {**API_PARAMS, "model_id": args.model}
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    if args.num_threads:
        NUM_THREADS = args.num_threads
    if args.num_gpus:
        NUM_GPUS = args.num_gpus

    # ---- Init GPU pool (distribute GPU IDs to threads) ----
    from multiprocessing import Manager
    gpu_pool = Manager().Queue()
    for gid in range(NUM_GPUS):
        gpu_pool.put(gid)

    # ---- Load few-shot example from arrow file ----
    arrow_path = "/ms/FM/gongoubo/new_project/slime_project/Data_Synthesis_For_Cuda_Kernel/ms_swift/data-00000-of-00001.arrow"
    ds = Dataset.from_file(arrow_path)
    example_idx = random.randrange(len(ds))
    example = ds[example_idx]
    ref_arch_torch = example["pytorch_code"]
    ref_arch_kernel = example["generated_code"]
    print(f"Few-shot: idx={example_idx}, uuid={example['uuid']}, problem={example['problem_name']}")

    # ---- Load KernelBench tasks ----
    tasks = load_kernelbench_tasks()
    print(f"Loaded {len(tasks)} KernelBench problems ({', '.join(f'{l}: {sum(1 for t in tasks if t[0]==l)}' for l in KB_LEVELS)})")

    # ---- Setup output ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    print(f"Processing with {NUM_THREADS} threads, {NUM_GPUS} GPU(s)")

    # ---- Run with ThreadPoolExecutor ----
    results = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(
                process_one_sample,
                level, problem_name, code, ref_arch_torch, ref_arch_kernel, OUTPUT_DIR,
            ): (level, problem_name)
            for level, problem_name, code, _ in tasks
        }

        for future in as_completed(futures):
            level, name = futures[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"Task failed for {level}/{name}: {e}")
                traceback.print_exc()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Done. Processed {len(results)}/{len(tasks)} samples.")
    if results:
        speedups = [r["speedup"] for r in results if r.get("speedup", 0) > 0]
        corrects = sum(1 for r in results if r.get("correctness", False))
        # By level
        by_level = {}
        for r in results:
            lv = r.get("level", "?")
            by_level.setdefault(lv, {"total": 0, "correct": 0})
            by_level[lv]["total"] += 1
            if r.get("correctness"):
                by_level[lv]["correct"] += 1
        print("By level:")
        for lv in KB_LEVELS:
            if lv in by_level:
                s = by_level[lv]
                print(f"  {lv}: {s['total']} processed, {s['correct']} correct ({s['correct']/max(s['total'],1)*100:.1f}%)")
        if speedups:
            print(f"Avg speedup (non-zero): {sum(speedups)/len(speedups):.2f}x")
            print(f"Max speedup: {max(speedups):.2f}x")
        print(f"Total correct: {corrects}/{len(results)}")
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
