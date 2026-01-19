# Token ID Caching Fix for AgentExecutionEngine

## Overview

This document explains the token ID caching mechanism implemented to eliminate retokenization in `AgentExecutionEngine` during multi-step agent-environment interactions.

## Problem

In the original implementation, `VerlEngine.get_model_response()` retokenized the entire prompt at each step:

```python
# OLD CODE (verl_engine.py:62-63)
prompt = self.chat_parser.parse(messages, ...)  # Convert all messages to text
request_prompt_ids = self.tokenizer.encode(prompt, ...)  # Retokenize everything!
```

In multi-step interactions, this caused:
1. **Redundant tokenization**: Previous completions (already tokenized by vLLM) were retokenized
2. **Token mismatches**: Retokenization might produce different token IDs than vLLM's original output
3. **Performance overhead**: Unnecessary tokenization of large prompts at each step

## Solution: Incremental Token ID Caching

### Architecture

The fix implements an incremental caching mechanism:

```
Step 0: Tokenize initial prompt → prompt_ids_0
        vLLM generates → completion_ids_0
        Cache: prompt_ids_0 + completion_ids_0

Step 1: Use cached tokens + tokenize only new env messages → prompt_ids_1
        vLLM generates → completion_ids_1  
        Cache: prompt_ids_1 + completion_ids_1

Step N: Use cached tokens + tokenize only new env messages → prompt_ids_N
        vLLM generates → completion_ids_N
        Cache: prompt_ids_N + completion_ids_N
```

### Implementation Details

#### 1. VerlEngine Modifications

**File**: `rllm/engine/rollout/verl_engine.py`

Added parameters to `get_model_response()`:
- `cached_prompt_ids`: Token IDs from previous steps (prompt + completions)
- `num_cached_messages`: Number of messages already represented in the cache

```python
async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
    cached_prompt_ids = kwargs.pop("cached_prompt_ids", None)
    num_cached_messages = kwargs.pop("num_cached_messages", 0)
    
    if cached_prompt_ids is not None and num_cached_messages > 0:
        # Only tokenize new messages
        new_messages = messages[num_cached_messages:]
        if new_messages:
            new_prompt = self.chat_parser.parse(new_messages, ...)
            new_token_ids = self.tokenizer.encode(new_prompt, ...)
            request_prompt_ids = cached_prompt_ids + new_token_ids
        else:
            request_prompt_ids = cached_prompt_ids
    else:
        # First step: tokenize everything
        prompt = self.chat_parser.parse(messages, ...)
        request_prompt_ids = self.tokenizer.encode(prompt, ...)
```

#### 2. AgentExecutionEngine Modifications

**File**: `rllm/engine/agent_execution_engine.py`

Added token ID caching state:

```python
# Track accumulated token IDs across steps
accumulated_token_ids = []
num_cached_messages = 0
```

Updated model response call to pass cache:

```python
for step_idx in range(self.max_steps):
    prompt_messages = agent.chat_completions.copy()
    
    # Pass cache for steps after the first
    if step_idx > 0 and accumulated_token_ids:
        kwargs["cached_prompt_ids"] = accumulated_token_ids.copy()
        kwargs["num_cached_messages"] = num_cached_messages
    
    model_output = await self.get_model_response(prompt_messages, ...)
    
    # Update cache with this step's tokens
    accumulated_token_ids = model_output.prompt_ids + model_output.completion_ids
    num_cached_messages = len(prompt_messages) + 1
```

### Token Flow Comparison

#### Before (with retokenization):
```
Step 0: messages[0] → tokenize → [1,2,3] → vLLM → [4,5]
Step 1: messages[0,1,2] → tokenize → [1,2,3,6,7,8,9] → vLLM → [10,11]
        ❌ [6,7,8,9] should be [4,5,6,7] but retokenization changed it!
```

#### After (with caching):
```
Step 0: messages[0] → tokenize → [1,2,3] → vLLM → [4,5]
        Cache: [1,2,3,4,5]
        
Step 1: cache[1,2,3,4,5] + tokenize(messages[2]) → [1,2,3,4,5,6,7] → vLLM → [8,9]
        ✅ [4,5] preserved from vLLM output, only new message tokenized
        Cache: [1,2,3,4,5,6,7,8,9]
```

## Benefits

1. **Zero Retokenization of Completions**: vLLM-generated token IDs are never retokenized
2. **Reduced Token Mismatches**: Only new environment messages are tokenized, existing tokens reused
3. **Performance**: Avoids retokenizing large conversation histories
4. **Correctness**: Training uses token IDs closer to what vLLM actually generated

## Limitations

1. **New Messages Still Tokenized**: Environment observations are still tokenized locally (not from vLLM)
2. **Not 100% Perfect**: Only completions from vLLM are guaranteed to match; new user/tool messages use local tokenizer
3. **Partial Solution**: For 100% guarantee, use `AgentSdkEngine` which captures all token IDs via traces

## When to Use

### Use AgentExecutionEngine with Token Caching When:
- ✅ You have existing code using `AgentExecutionEngine`
- ✅ You want to reduce retokenization without major refactoring
- ✅ Minor token mismatches for environment messages are acceptable

### Use AgentSdkEngine When:
- ✅ You need 100% token ID fidelity
- ✅ You're starting a new project
- ✅ You can use the SDK workflow with proxy and traces

## Testing

To verify the fix works:

1. **Check logs**: Token mismatch warnings should decrease significantly
2. **Monitor metrics**: `token_mismatch` metric in training should be lower
3. **Inspect episode_steps**: `prompt_ids` in consecutive steps should have proper overlap

## Configuration

No configuration changes needed. The caching is enabled automatically for all `AgentExecutionEngine` users.

To disable (for debugging):
- Caching only activates when `step_idx > 0`
- First step always tokenizes from scratch

## Backward Compatibility

✅ **Fully backward compatible**
- No API changes required
- Existing training scripts work without modification
- Fallback to full tokenization if cache parameters not provided

## Performance Impact

- **Memory**: Minimal (stores ~10-100KB per trajectory for token ID lists)
- **Speed**: Faster (avoids retokenizing large prompts)
- **Training**: Potentially better (fewer token mismatches)

## Future Work

For complete elimination of retokenization:
1. Capture token IDs for environment messages from vLLM
2. Implement full trace storage like `AgentSdkEngine`
3. Add configuration to choose between caching strategies
