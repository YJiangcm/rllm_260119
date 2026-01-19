# Quick Reference: Token ID Flow in rLLM

## One-Line Summary

rLLM's training pipelines capture token IDs directly from vLLM's inference engine and use them as-is for training - no retokenization ever occurs.

**Both `AgentSdkEngine` and `AgentExecutionEngine` guarantee zero retokenization.**

## Engine Comparison

| Engine | Retokenization? | Token ID Source | Use Case |
|--------|----------------|-----------------|----------|
| **AgentSdkEngine** ✅ | ❌ No | vLLM (via traces) | SDK workflows, complex programs |
| **AgentExecutionEngine** ✅ | ❌ No | vLLM (via episode_steps) | Agent-environment RL training |

Both engines use vLLM token IDs directly for training without retokenization.

## Flow Diagram

### AgentSdkEngine Pipeline

```
vLLM Engine → LiteLLM Proxy → SQLite Trace Store → Training Pipeline
  (generates)   (captures)        (persists)         (uses directly)
  token_ids     token_ids         token_ids          token_ids
     ↓              ↓                 ↓                  ↓
  [1,2,3]       [1,2,3]           [1,2,3]            [1,2,3]
```

### AgentExecutionEngine Pipeline

```
vLLM Engine → ModelOutput → episode_steps → assemble_steps() → Training
  (generates)   (captures)     (stores)         (uses)           (uses)
  token_ids     token_ids      token_ids        token_ids        token_ids
     ↓              ↓              ↓                ↓                ↓
  [1,2,3]       [1,2,3]        [1,2,3]          [1,2,3]          [1,2,3]
```

**Both pipelines**: ✅ NO TEXT! ✅ NO TOKENIZE! ✅ TOKEN IDs ARE IMMUTABLE!

## Key Files

### AgentSdkEngine Pipeline

| Component | File | Purpose |
|-----------|------|---------|
| Capture | `rllm/patches/vllm_instrumentation.py` | Extract token IDs from vLLM |
| Protocol | `rllm/sdk/protocol.py` | Define Trace structure with token IDs |
| Extract | `rllm/sdk/data_process.py` | Convert Trace → ModelOutput |
| Store | `rllm/sdk/proxy/litellm_callbacks.py` | Persist traces to SQLite |
| Train | `rllm/engine/agent_sdk_engine.py` | Transform to training format |

### AgentExecutionEngine Pipeline

| Component | File | Purpose |
|-----------|------|---------|
| Capture | `rllm/engine/rollout/verl_engine.py` | Get ModelOutput with token IDs from vLLM |
| Store | `rllm/engine/agent_execution_engine.py:252-253` | Store in episode_steps |
| Assemble | `rllm/engine/agent_execution_engine.py:446-481` | assemble_steps() uses vLLM token IDs |
| Train | `rllm/trainer/verl/agent_ppo_trainer.py` | Use token data for training |

**Note**: `convert_messages_to_tokens_and_masks()` in AgentExecutionEngine is only for runtime token counting, NOT for training data.

## Data Structures

```python
# 1. vLLM Response (captured)
{
    "prompt_token_ids": [1, 2, 3],        # Prompt
    "choices": [{
        "token_ids": [4, 5, 6],           # Completion
        "response_logprobs": [0.5, -1.2]  # Optional
    }]
}

# 2. Trace (stored)
Trace(
    input=LLMInput(
        messages=[...],
        prompt_token_ids=[1, 2, 3]        # ← From vLLM
    ),
    output=LLMOutput(
        message={...},
        output_token_ids=[4, 5, 6],       # ← From vLLM
        rollout_logprobs=[0.5, -1.2]
    )
)

# 3. ModelOutput (training)
ModelOutput(
    prompt_ids=[1, 2, 3],                 # ← From Trace
    completion_ids=[4, 5, 6],             # ← From Trace
    logprobs=[0.5, -1.2]
)

# 4. PyTorch Tensors (training)
input_ids = torch.tensor([1, 2, 3, 4, 5, 6])  # ← Direct conversion
```

## Why No Retokenization?

### Both Pipelines Guarantee Zero Retokenization

1. **vLLM is source**: Token IDs come from `RequestOutput.prompt_token_ids` and `output.token_ids`
2. **Direct storage**: 
   - AgentSdkEngine: Stored in SQLite Trace
   - AgentExecutionEngine: Stored in episode_steps
3. **Direct pipeline**: `list[int]` → storage → training (pure data transforms)
4. **No text used for training**: Text is for debugging/runtime checks only

### Common Misconception

`AgentExecutionEngine` calls `convert_messages_to_tokens_and_masks()`, but this is **only for runtime token counting** (checking if prompts/responses exceed length limits). The actual training data comes from `episode_steps` which contains the original vLLM token IDs stored at lines 252-253.

## Quick Setup

### For vLLM < 0.10.2

```python
from rllm.patches.vllm_instrumentation import instrument_vllm

# BEFORE creating AgentLoopManager
instrument_vllm(add_response_logprobs=True)
```

### For vLLM >= 0.10.2

```python
# Native support - just configure proxy
proxy_config = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "add_logprobs": True,  # Enable logprobs capture
}
```

## Verification

```python
from rllm.patches.vllm_instrumentation import get_vllm_token_ids_support

support = get_vllm_token_ids_support()
# Returns: "native" | "instrumented" | "none" | "unavailable"
print(f"Token ID support: {support}")
```

## Benefits

✅ **100% Fidelity** - Exact tokens from inference used in training  
✅ **No Tokenization Bugs** - Eliminates mismatch between rollout and training  
✅ **Performance** - No redundant tokenization overhead  
✅ **Correctness** - Special tokens handled identically in inference and training  
✅ **Debuggable** - Token IDs in traces for easy inspection  

## Read More

See [Token ID Capture Flow](token_id_capture_flow.md) for detailed technical documentation.
