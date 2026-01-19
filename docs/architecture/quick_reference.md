# Quick Reference: Token ID Flow in rLLM

## One-Line Summary

rLLM's **SDK-based pipeline** (`AgentSdkEngine`) captures token IDs directly from vLLM's inference engine and uses them as-is for training - no retokenization ever occurs.

> **⚠️ Important**: `AgentExecutionEngine` retokenizes prompts at each step. Only `AgentSdkEngine` guarantees zero retokenization.

## Engine Comparison

| Engine | Prompt Token IDs | Completion Token IDs | Zero Retokenization? |
|--------|------------------|---------------------|---------------------|
| **AgentSdkEngine** ✅ | From vLLM (via traces) | From vLLM (via traces) | ✅ Yes |
| **AgentExecutionEngine** ⚠️ | Retokenized locally | From vLLM | ❌ No (prompts retokenized) |

## Flow Diagram

### AgentSdkEngine Pipeline ✅ Zero Retokenization

```
vLLM Engine → LiteLLM Proxy → SQLite Trace Store → Training Pipeline
  (generates)   (captures)        (persists)         (uses directly)
  token_ids     token_ids         token_ids          token_ids
     ↓              ↓                 ↓                  ↓
  [1,2,3]       [1,2,3]           [1,2,3]            [1,2,3]
```

### AgentExecutionEngine Pipeline ⚠️ Prompts Retokenized

```
vLLM Engine → VerlEngine → episode_steps → assemble_steps() → Training
  (generates)  (RETOKENIZE     (stores        (detects           (uses)
  token_ids     prompts!)      mixed IDs)      mismatches)       mixed IDs)
     ↓              ↓               ↓              ↓                ↓
  [1,2,3]      [1,2,3] ⚠️      [1,2,3] ⚠️    mask if invalid   [1,2,3] ⚠️
               (prompt)         (prompt)                        (prompt)
               [4,5,6] ✅      [4,5,6] ✅                       [4,5,6] ✅
               (completion)     (completion)                    (completion)
```

**AgentSdkEngine**: ✅ NO RETOKENIZATION - Both prompts and completions from vLLM  
**AgentExecutionEngine**: ⚠️ PROMPTS RETOKENIZED - Only completions from vLLM

## Key Files

### AgentSdkEngine Pipeline (Zero Retokenization) ✅

| Component | File | Purpose |
|-----------|------|---------|
| Capture | `rllm/patches/vllm_instrumentation.py` | Extract token IDs from vLLM |
| Protocol | `rllm/sdk/protocol.py` | Define Trace structure with token IDs |
| Extract | `rllm/sdk/data_process.py` | Convert Trace → ModelOutput |
| Store | `rllm/sdk/proxy/litellm_callbacks.py` | Persist traces to SQLite |
| Train | `rllm/engine/agent_sdk_engine.py` | Transform to training format |

### AgentExecutionEngine Pipeline (Prompts Retokenized) ⚠️

| Component | File | Purpose |
|-----------|------|---------|
| Capture | `rllm/engine/rollout/verl_engine.py:81` | Get completion IDs from vLLM ✅ |
| **Retokenize** | `rllm/engine/rollout/verl_engine.py:62-63` | **Retokenize prompts** ⚠️ |
| Store | `rllm/engine/agent_execution_engine.py:252-253` | Store mixed token IDs in episode_steps |
| Detect | `rllm/engine/agent_execution_engine.py:462-477` | Detect token mismatches in assemble_steps() |
| Mask | `rllm/engine/agent_execution_engine.py:489-490` | Mask invalid trajectories |
| Train | `rllm/trainer/verl/agent_ppo_trainer.py` | Use mixed token IDs for training ⚠️ |

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

### AgentSdkEngine ✅

1. **vLLM is source**: Both prompt and completion token IDs come from vLLM
2. **Immutable storage**: Stored in SQLite Trace, never modified
3. **Direct pipeline**: `list[int]` → `Trace` → `ModelOutput` → `tensor` (pure data transforms)
4. **No text used for training**: Text is for debugging only

### AgentExecutionEngine ⚠️

**Completions**: ✅ From vLLM (`token_output.token_ids`)  
**Prompts**: ❌ Retokenized by local tokenizer

```python
# In VerlEngine.get_model_response() (verl_engine.py:62-63):
prompt = self.chat_parser.parse(messages, ...)  # Convert to text
request_prompt_ids = self.tokenizer.encode(prompt, ...)  # RETOKENIZE!
```

**Detection & Mitigation**:
- `assemble_steps()` detects when token sequences don't align across steps
- Warns: _"This is likely due to retokenization"_ (line 472)
- Sets `is_valid_trajectory = False`
- If `config.rllm.filter_token_mismatch = True`, masks the trajectory

**For guaranteed zero retokenization, use `AgentSdkEngine`**.

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
