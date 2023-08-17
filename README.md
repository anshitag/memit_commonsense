# Editing Common Sense in Transformers

Editing commonsense judgments using causally associated localized, editable parameters in Transformers

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Base Finetuning](#base-finetuning)
- [Causal Tracing](#causal-tracing)
- [Repair Finetuning](#repair-finetuning)
- [MEMIT_CSK Experiment](#memit_csk-experiment)
- [How to Cite](#how-to-cite)

## Installation

Similar to [MEMIT](https://github.com/kmeng01/memit) installation instructions. 
We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`.

## Datasets
The 20 Question and PEP 3K datasets are under the [data](data) folder. Each dataset consists of Train, Edit, Edit Validation and Probe Sets. 

They also contain incorrect subsets used for editing using $MEMIT_{CSK}$.

## Base Finetuning

[`script_base_finetuning.sh`](base_finetune_experiments/script_base_finetuning.sh) can be used for running base finetuning on the gpt2-xl model for the 20q dataset. Similar command can be used for running experiments for gpt2-large model and PEP 3K dataset.

## Causal Tracing

[`script_causal_trace_zero_shot.sh`](causal_tracing_experiment/script_causal_trace_zero_shot.sh) can be used for performing causal tracing experiment for zero shot model.

[`script_causal_trace.sh`](causal_tracing_experiment/script_causal_trace.sh) can be used for performing causal tracing experiment for base fintuned model, by passing it's checkpoint location as a parameter and it's output inference file.

[`script_causal_trace_severed.sh`](causal_tracing_experiment/script_causal_trace_severed.sh) can be used for performing *severed* causal tracing experiment for base fintuned model, by passing it's checkpoint location as a parameter and it's output inference file.

## Repair Finetuning

[`script_repair_finetuning.sh`](repair_finetune_experiments/script_repair_finetuning.sh) can be used for running repair finetuning on the base finetuned gpt2-xl model for the 20q dataset. Similar command can be used for running experiments for gpt2-large model and PEP 3K dataset. 

It includes commands to evaluate the affected and unaffected metrics for the repair finetuned model.

## MEMIT_CSK Experiment

[`script_memit_csk.sh`](script_memit_csk.sh) can be used for running $MEMIT_{CSK}$ on the base finetuned gpt2-xl model for the 20q dataset. Similar commands can be used for running experiments for gpt2-large model and PEP 3K dataset. 

It includes commands to first find best hyperparamters for the Edit Validation Set, i.e. *configuration generalization*, followed by evaluation on Edit Set.
Then command for running evalution on Probe Set, i.e. *semantic generalization*.

## How to Cite

```bibtex
@article{gupta2023editing,
  title={Editing Commonsense Knowledge in GPT},
  author={Gupta, Anshita and Mondal, Debanjan and Sheshadri, Akshay Krishna and Zhao, Wenlong and Li, Xiang Lorraine and Wiegreffe, Sarah and Tandon, Niket},
  journal={arXiv preprint arXiv:2305.14956},
  year={2023}
}
```
