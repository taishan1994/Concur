import pandas as pd
import random
from datasets import Dataset

# Load the arrow file
ds = Dataset.from_file('data-00000-of-00001.arrow')

template = """You are a Machine Learning Engineer trying to write custom cuda kernels to replace the pytorch operators in the given architecture to get speedups. You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom cuda kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes ( such as online softmax). You are only limited by your imagination. For [Imports], you will likely need but not limited to the following libraries:
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

# Pick a random sample as the few-shot example, exclude it from training data
example_idx = random.randrange(len(ds))
example = ds[example_idx]
ref_arch_torch = example['pytorch_code']
ref_arch_kernel = example['generated_code']

print(f"Using sample {example_idx} (uuid={example['uuid']}, problem={example['problem_name']}) as few-shot example")

records = []
for i, row in enumerate(ds):
    if i == example_idx:
        continue

    user_content = template.replace('$ref_arch_torch', ref_arch_torch).replace('$ref_arch_kernel', ref_arch_kernel).replace('$code', row['pytorch_code'])
    assistant_content = f"<think>\n{row['generated_think']}\n</think>\n{row['generated_code']}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    records.append({
        "messages": messages,
        "uuid": row['uuid'],
        "problem_name": row['problem_name'],
        "speedup_over_eager": row['speedup_over_eager'],
    })

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(ds)} rows")

# Save as parquet
df_out = pd.DataFrame(records)
df_out.to_parquet('dataset_sft_from_arrow.parquet', index=False)
print(f"Done! Saved {len(df_out)} rows to dataset_sft_from_arrow.parquet (excluded example {example_idx})")
