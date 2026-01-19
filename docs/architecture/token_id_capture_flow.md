# Token ID Capture Flow: Direct Integration from vLLM to Training

## Overview

This document explains how rLLM captures token IDs directly from the vLLM inference server, stores them, and uses them for training **without any retokenization**. This design ensures 100% fidelity between inference and training tokens.

## Why This Matters

**Problem**: Traditional RL training pipelines often:
1. Generate text from a model
2. Store the text
3. Re-tokenize the text for training

This introduces tokenization mismatches that can harm training quality.

**Solution**: rLLM captures token IDs directly from vLLM's internal engine and uses them as-is for training, eliminating retokenization entirely.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TOKEN ID FLOW PIPELINE                           │
│                                                                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │  vLLM    │───▶│  Proxy   │───▶│  Trace   │───▶│ Training │     │
│  │ Engine   │    │ Manager  │    │  Store   │    │ Pipeline │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│       │               │               │               │              │
│   token_ids      capture IDs      store IDs      use IDs           │
│  (no text!)     (no tokenize!)   (immutable!)   (no tokenize!)     │
└─────────────────────────────────────────────────────────────────────┘
```

## Detailed Flow

### 1. Token ID Capture from vLLM

**Location**: `rllm/patches/vllm_instrumentation.py`

#### For vLLM < 0.10.2

The system monkey-patches vLLM's OpenAI-compatible API to extract token IDs:

```python
# Lines 161-240: Patched generator intercepts vLLM's internal response
async def chat_completion_full_generator(...):
    async def _generate_interceptor():
        async for res in result_generator:
            yield res
            # Extract token IDs directly from vLLM's RequestOutput
            prompt_token_ids = res.prompt_token_ids  # Prompt tokens
            response_token_ids = [output.token_ids for output in res.outputs]  # Response tokens
            
            # Optional: Extract logprobs for each token
            if add_response_logprobs:
                response_logprobs_lists = [...]
    
    # Call original generator with interceptor
    response = await _original_chat_completion_full_generator(...)
    
    # Add token IDs to response in vLLM 0.10.2+ native format
    response.prompt_token_ids = prompt_token_ids  # Top level
    response.choices[i].token_ids = response_token_ids[i]  # Per choice
```

**Key Point**: Token IDs are extracted from `RequestOutput.prompt_token_ids` and `RequestOutput.outputs[i].token_ids` - these are the **actual tokens vLLM used internally**, not reconstructed from text.

#### For vLLM >= 0.10.2

No patching needed - use native `return_token_ids=True` parameter:

```python
# In litellm_callbacks.py:
data["return_token_ids"] = True  # Native vLLM support
```

#### Token ID Format

Response structure matches vLLM 0.10.2+ native format:

```json
{
  "prompt_token_ids": [123, 456, 789],
  "choices": [
    {
      "message": {"content": "...", "role": "assistant"},
      "token_ids": [111, 222, 333],
      "response_logprobs": [0.5, -1.2, 0.8]
    }
  ]
}
```

### 2. Storage in Traces

**Location**: `rllm/sdk/protocol.py`, `rllm/sdk/data_process.py`

#### Protocol Definition

```python
class LLMInput(BaseModel):
    messages: list[dict]
    prompt_token_ids: list[int]  # ← Captured from vLLM

class LLMOutput(BaseModel):
    message: dict
    finish_reason: str
    output_token_ids: list[int]  # ← Captured from vLLM
    rollout_logprobs: list[float] | None

class Trace(BaseModel):
    trace_id: str
    session_name: str
    input: LLMInput      # Contains prompt_token_ids
    output: LLMOutput    # Contains output_token_ids
    model: str
    latency_ms: float
    tokens: dict[str, int]
    metadata: dict
    timestamp: float
```

#### Extraction from Response

```python
# rllm/sdk/data_process.py

def _extract_prompt_token_ids(output_payload: dict) -> list[int]:
    """Extract prompt token IDs from response root level."""
    prompt_ids = output_payload.get("prompt_token_ids")
    return list(prompt_ids) if prompt_ids else []

def _extract_completion_token_ids(output_payload: dict) -> list[int]:
    """Extract completion token IDs from choices[0].provider_specific_fields."""
    completion_ids = output_payload.get("choices")[0].get(
        "provider_specific_fields", {}
    ).get("token_ids")
    return list(completion_ids) if completion_ids else []

def build_llm_io(input_payload: dict, output_payload: dict) -> tuple[LLMInput, LLMOutput]:
    """Normalize raw OpenAI input/output into structured LLMInput/LLMOutput."""
    prompt_token_ids = _extract_prompt_token_ids(output_payload)
    
    llm_input = LLMInput(
        messages=input_payload.get("messages", []),
        prompt_token_ids=prompt_token_ids  # ← Token IDs stored directly
    )
    
    completion_ids = _extract_completion_token_ids(output_payload)
    llm_output = LLMOutput(
        message=output_payload["choices"][0].get("message", {}),
        finish_reason=output_payload["choices"][0].get("finish_reason"),
        output_token_ids=completion_ids,  # ← Token IDs stored directly
        rollout_logprobs=_extract_logprobs(output_payload)
    )
    
    return llm_input, llm_output
```

#### Persistence via Proxy

**Location**: `rllm/sdk/proxy/litellm_callbacks.py`

```python
class TracingCallback(CustomLogger):
    """Log LLM calls to tracer with token IDs."""
    
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: ModelResponse,
    ):
        # Extract full response payload (includes token IDs)
        response_payload = response.model_dump()
        
        # Log to SQLite via tracer
        await self.tracer.log_llm_call(
            name=f"proxy/{model}",
            model=model,
            input={"messages": messages},
            output=response_payload,  # ← Contains prompt_token_ids and token_ids
            metadata=metadata,
            ...
        )
```

**Critical**: The response payload stored contains **both text AND token IDs**. Training uses token IDs; text is for debugging/visualization only.

### 3. Extraction for Training

**Location**: `rllm/sdk/data_process.py`, `rllm/engine/agent_sdk_engine.py`

#### Trace → ModelOutput Conversion

```python
def trace_to_model_output(trace: Trace) -> ModelOutput:
    """Convert stored Trace to ModelOutput for training."""
    prompt_ids = trace.input.prompt_token_ids      # ← Direct extraction
    completion_ids = trace.output.output_token_ids  # ← Direct extraction
    
    assert prompt_ids, "Prompt IDs are required"
    assert completion_ids, "Completion IDs are required"
    
    return ModelOutput(
        text="",  # Text not used for training
        content=trace.output.message.get("content", ""),
        reasoning=trace.output.message.get("reasoning", ""),
        tool_calls=trace.output.message.get("tool_calls", []),
        prompt_ids=prompt_ids,          # ← Pure token ID pass-through
        completion_ids=completion_ids,  # ← Pure token ID pass-through
        logprobs=trace.output.rollout_logprobs or [],
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
        finish_reason=trace.output.finish_reason or "stop",
    )
```

#### Trace → Step → Trajectory

```python
def trace_to_step(trace: Trace) -> Step:
    """Convert Trace to training Step."""
    return Step(
        chat_completions=trace.input.messages + [trace.output.message],
        model_output=trace_to_model_output(trace),  # ← Contains token IDs
        info=trace.metadata,
    )

# In agent_sdk_engine.py:
for trace in all_traces:
    trace_obj = Trace(**trace.data)
    step = trace_to_step(trace_obj)
    steps.append(step)

trajectories = group_steps(steps, by=self.groupby_key)
```

#### Transform for VERL Training

**Location**: `rllm/engine/agent_sdk_engine.py:459-707`

```python
def transform_results_for_verl(self, episodes: list[Episode]) -> DataProto:
    """Transform episodes to VERL-compatible format."""
    prompts = []
    responses = []
    rollout_logprobs = []
    
    for episode in episodes:
        for trajectory in episode.trajectories:
            for step in trajectory.steps:
                if isinstance(step.model_output, ModelOutput):
                    # Use token IDs directly - no tokenization!
                    prompt_ids = torch.tensor(
                        step.model_output.prompt_ids,  # ← Direct use
                        dtype=torch.long
                    )
                    prompts.append(prompt_ids)
                    
                    response_ids = torch.tensor(
                        step.model_output.completion_ids,  # ← Direct use
                        dtype=torch.long
                    )
                    responses.append(response_ids)
                    
                    logprobs = torch.tensor(
                        step.model_output.logprobs,  # ← Direct use
                        dtype=torch.float32
                    )
                    rollout_logprobs.append(logprobs)
    
    # Pad sequences for batching (still no tokenization!)
    prompts_batch = torch.nn.utils.rnn.pad_sequence(prompts, ...)
    response_batch = torch.nn.utils.rnn.pad_sequence(responses, ...)
    
    # Concatenate for training
    input_ids = torch.concat([prompts_batch, response_batch], dim=1)
    
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,  # ← Pure token IDs, no retokenization
            "responses": response_batch,
            "rollout_log_probs": rollout_logprobs_batch,
            ...
        }
    )
```

### 4. Training Pipeline (Tinker)

**Location**: `rllm/trainer/tinker/tinker_data_processor.py`

```python
def build_datum_from_step(step: Step, advantage: float) -> tinker.Datum:
    """Build Tinker training datum from step."""
    # Extract token IDs directly from ModelOutput
    prompt_tokens = step.prompt_ids      # ← No tokenization
    response_tokens = step.response_ids  # ← No tokenization
    logprobs = step.logprobs
    
    # Concatenate prompt + response
    all_tokens = prompt_tokens + response_tokens  # ← Pure list concatenation
    input_tokens = all_tokens[:-1]   # Teacher forcing: input
    target_tokens = all_tokens[1:]   # Teacher forcing: target
    
    # Create training datum from raw token IDs
    datum = tinker.types.Datum(
        model_input=tinker.types.ModelInput.from_ints(
            tokens=input_tokens  # ← Tinker accepts pre-tokenized IDs
        ),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs)),
            "advantages": TensorData.from_torch(torch.tensor(advantages)),
            "mask": TensorData.from_torch(torch.tensor(mask)),
        }
    )
    
    return datum
```

**Key Point**: `ModelInput.from_ints(tokens=...)` accepts raw token lists. No tokenizer is called.

## Key Integration Points

| Stage | Location | Input | Output | Tokenization? |
|-------|----------|-------|--------|---------------|
| **Capture** | `vllm_instrumentation.py:184` | vLLM `RequestOutput` | `prompt_token_ids`, `token_ids` | ❌ Already done by vLLM |
| **Store** | `litellm_callbacks.py:54` | API response dict | Trace with token IDs | ❌ Pure storage |
| **Extract** | `data_process.py:13-28` | Trace payload | `list[int]` | ❌ Field extraction only |
| **Convert** | `data_process.py:91-118` | Trace | ModelOutput | ❌ Pure data transform |
| **Transform** | `agent_sdk_engine.py:554-567` | ModelOutput | PyTorch tensors | ❌ `torch.tensor(list[int])` |
| **Train** | `tinker_data_processor.py:215` | Token ID lists | Tinker Datum | ❌ `from_ints()` direct |

## Why No Retokenization Occurs

1. **vLLM is Source of Truth**: Token IDs come directly from vLLM's tokenization engine via `RequestOutput.prompt_token_ids` and `RequestOutput.outputs[i].token_ids`

2. **Immutable Storage**: Once captured, token IDs are stored in `Trace.input.prompt_token_ids` and `Trace.output.output_token_ids` and never modified

3. **Direct Pipeline**: The entire flow is pure data transformation:
   ```
   list[int] → Trace → ModelOutput → Step → Tensor → Datum
   ```
   No tokenizer is ever called after vLLM generates tokens.

4. **No Text Reconstruction**: Text content exists only in `Trace.output.message["content"]` for debugging/visualization. Training uses `output_token_ids` exclusively.

5. **Framework Integration**: Both VERL and Tinker accept pre-tokenized inputs:
   - VERL: `input_ids` tensor is built from `step.model_output.prompt_ids` + `step.model_output.completion_ids`
   - Tinker: `ModelInput.from_ints(tokens=...)` accepts raw token lists

## Configuration

### Enable Token ID Capture

For vLLM < 0.10.2, instrument before creating VERL engine:

```python
from rllm.patches.vllm_instrumentation import instrument_vllm

# IMPORTANT: Call BEFORE creating AgentLoopManager
instrument_vllm(add_response_logprobs=True)

# Then create VERL engine
config = ...
rollout_manager = AgentLoopManager(config)
verl_engine = VerlEngine(config, rollout_manager, tokenizer)
```

For vLLM >= 0.10.2, enable via proxy config:

```python
proxy_config = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "add_logprobs": True,
    # No instrumentation needed - native support
}

engine = AgentSdkEngine(
    agent_run_func=agent_func,
    rollout_engine=verl_engine,
    proxy_config=proxy_config,
    ...
)
```

### Verify Token IDs

```python
from rllm.patches.vllm_instrumentation import (
    get_vllm_token_ids_support,
    check_vllm_instrumentation_status
)

# Check support level
support = get_vllm_token_ids_support()
# Returns: "native" | "instrumented" | "none" | "unavailable"

# Get detailed status
status = check_vllm_instrumentation_status()
print(f"vLLM version: {status['vllm_version']}")
print(f"Instrumented: {status['is_instrumented_flag']}")
```

## Benefits

1. **100% Fidelity**: Training uses the exact tokens vLLM generated during inference
2. **No Tokenization Bugs**: Eliminates tokenization mismatches between rollout and training
3. **Performance**: No redundant tokenization overhead
4. **Correctness**: Special tokens, unknown tokens, and edge cases handled identically in inference and training
5. **Debugging**: Token IDs in traces make it easy to inspect exact model behavior

## Implementation Notes

### Ray Workers and Instrumentation

**Important**: vLLM instrumentation must be applied **before** VERL creates Ray workers. Monkey patches in the main process do NOT propagate to Ray worker processes.

```python
# ✅ CORRECT
instrument_vllm()  # Before AgentLoopManager
rollout_manager = AgentLoopManager(config)

# ❌ WRONG
rollout_manager = AgentLoopManager(config)
instrument_vllm()  # Too late! Workers already started
```

### Storage Format

Token IDs are stored in `provider_specific_fields` within the response:

```python
response = {
    "prompt_token_ids": [1, 2, 3],  # Top level
    "choices": [{
        "message": {...},
        "provider_specific_fields": {
            "token_ids": [4, 5, 6],  # Completion tokens
            "response_logprobs": [0.5, -1.2, 0.8]  # Optional
        }
    }]
}
```

This format matches vLLM 0.10.2+ native format and is forward-compatible.

## See Also

- [SDK Overview](../core-concepts/sdk.md)
- [Training Configuration](../api/trainer/)
- [vLLM Documentation](https://docs.vllm.ai/)
