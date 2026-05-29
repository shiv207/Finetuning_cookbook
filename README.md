# Fine-tuning GPT-OSS (20B) for Agentic Tool Use

This repository documents a parameter-efficient fine-tuning run on OpenAI’s open-weight **gpt-oss-20B** model. The goal is to improve structured reasoning and tool-calling behavior using agent-oriented instruction data, while keeping training feasible on a single consumer GPU.

## Overview

Large language models are often strong at open-ended dialogue but weaker at following multi-step plans, emitting valid tool calls, and separating internal reasoning from final answers. **gpt-oss** addresses part of this through the [Harmony](https://github.com/openai/harmony) conversation format, which supports dedicated channels (for example `analysis` and `final`) and native tool-call syntax.

We fine-tuned [unsloth/gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b) with LoRA adapters on examples from [internlm/Agent-FLAN](https://huggingface.co/datasets/internlm/Agent-FLAN), focusing on the ReAct-style subset (`agent_instruct_react.jsonl`). Training was done with [Unsloth](https://github.com/unslothai/unsloth) on a Tesla T4 (Google Colab). The resulting weights are published as [shiv207/gpt_oss_AGENTBOI](https://huggingface.co/shiv207/gpt_oss_AGENTBOI).

## Approach

**Base model.** gpt-oss-20B is a mixture-of-experts (MoE) architecture released with open weights. Unsloth provides optimized training paths for this family, including MoE-aware LoRA targeting.

**Data.** Agent-FLAN contains agent instructions with conversation histories suited to tool use and step-by-step reasoning. We map each example into Harmony-compatible prompts so the model learns the expected channel structure and tool-call format.

**Training.** We use LoRA rather than full fine-tuning so only a small fraction of parameters are updated. This keeps memory use within the limits of a T4 while still adapting behavior to the agentic task distribution. Hyperparameters and the full pipeline (data prep, training, inference, export) are in `gpt_oss_(20B)_Fine_tuning.ipynb`.

**Inference.** At generation time, gpt-oss exposes a `reasoning_effort` setting (`low`, `medium`, `high`) that trades latency for depth of internal reasoning. The notebook includes examples at each level.

## Outcomes

After fine-tuning, the model more consistently:

- Uses Harmony channels for reasoning vs. user-facing output  
- Produces structured JSON tool calls (e.g. search and retrieval style actions)  
- Follows multi-step agent instructions in the ReAct-style evaluation examples in the notebook  

These are qualitative observations from the notebook’s inference cells, not a formal benchmark suite.

## Repository contents

| File | Description |
|------|-------------|
| `gpt_oss_(20B)_Fine_tuning.ipynb` | End-to-end Colab-oriented workflow: install, dataset prep, LoRA training, inference, and model upload |
| `README.md` | Project summary (this file) |

## Reproducing the work

1. Open `gpt_oss_(20B)_Fine_tuning.ipynb` in Google Colab (or a local environment with a compatible GPU and the dependencies listed in the notebook).  
2. Run the cells in order. The notebook is derived from Unsloth’s gpt-oss fine-tuning template and extended for Agent-FLAN.  
3. For a full training run, use `num_train_epochs=1` and disable the short `max_steps` cap used in the quick demo.

Published adapters can be loaded from Hugging Face with Unsloth’s `FastLanguageModel.from_pretrained("shiv207/gpt_oss_AGENTBOI")`; see the notebook for tokenizer settings and `reasoning_effort` usage.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient gpt-oss training tooling  
- [InternLM](https://huggingface.co/internlm) for the Agent-FLAN dataset  
- [OpenAI](https://openai.com/) for the gpt-oss open-weight release and Harmony format documentation  

## Author

shiv207
