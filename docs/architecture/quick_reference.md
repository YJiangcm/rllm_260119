# Quick Reference: Token ID Flow in rLLM

## One-Line Summary

rLLM captures token IDs directly from vLLM's inference engine and uses them as-is for training - no retokenization ever occurs.

## Flow Diagram

```
vLLM Engine → LiteLLM Proxy → SQLite Trace Store → Training Pipeline
  (generates)   (captures)        (persists)         (uses directly)
  token_ids     token_ids         token_ids          token_ids
     ↓              ↓                 ↓                  ↓
  [1,2,3]       [1,2,3]           [1,2,3]            [1,2,3]
     ↓              ↓                 ↓                  ↓
  NO TEXT!      NO TOKENIZE!      IMMUTABLE!        NO TOKENIZE!
```

## Key Files

| Component | File | Purpose |
|-----------|------|---------|
| Capture | `rllm/patches/vllm_instrumentation.py` | Extract token IDs from vLLM |
| Protocol | `rllm/sdk/protocol.py` | Define Trace structure with token IDs |
| Extract | `rllm/sdk/data_process.py` | Convert Trace → ModelOutput |
| Store | `rllm/sdk/proxy/litellm_callbacks.py` | Persist traces to SQLite |
| Train | `rllm/engine/agent_sdk_engine.py` | Transform to training format |

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

1. **vLLM is source**: Token IDs come from `RequestOutput.prompt_token_ids` and `output.token_ids`
2. **Immutable storage**: Stored in Trace, never modified
3. **Direct pipeline**: `list[int]` → `Trace` → `ModelOutput` → `tensor` (pure data transforms)
4. **No text used**: Text is for debugging only; training uses token IDs exclusively

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
