# SWE Agent Training - SDK Migration

## Overview

This directory contains both traditional and SDK-based training scripts for the SWE (Software Engineering) agent.

## Files

### Traditional AgentExecutionEngine (has retokenization issue)
- **`train_deepswe_32b.sh`** - Original training script using `AgentExecutionEngine`
- **`train_deepswe_agent.py`** - Python training script with agent-environment loop

**Issue**: Prompts are retokenized at each step in `VerlEngine.get_model_response()`, causing potential token ID mismatches.

### SDK-based AgentSdkEngine (zero retokenization) ✅
- **`train_deepswe_32b_sdk.sh`** - New SDK-based training script
- **`train_deepswe_sdk.py`** - Python training script using traced LLM calls

**Benefit**: Token IDs from vLLM are captured via LiteLLM proxy and stored in SQLite traces, ensuring 100% fidelity without retokenization.

## Migration Guide

### Prerequisites

1. **Install dependencies**:
   ```bash
   pip install rllm litellm
   ```

2. **Prepare SWE dataset**:
   ```bash
   python prepare_swe_data.py
   ```

### Option 1: Quick Start with SDK (Recommended)

1. **Start LiteLLM Proxy** (in a separate terminal):
   ```bash
   # Create proxy config
   cat > proxy_config.yaml <<EOF
   model_list:
     - model_name: Qwen/Qwen3-32B
       litellm_params:
         model: vllm/Qwen/Qwen3-32B
         api_base: http://localhost:8000/v1
         return_token_ids: true
   EOF
   
   # Launch proxy
   python3 -m rllm.sdk.proxy.launch_litellm_proxy --config proxy_config.yaml --port 4000
   ```

2. **Start vLLM server** (in another terminal):
   ```bash
   python3 -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen3-32B \
       --port 8000 \
       --tensor-parallel-size 8
   ```

3. **Run SDK-based training**:
   ```bash
   bash train_deepswe_32b_sdk.sh
   ```

### Option 2: Continue with Traditional Approach

If you need to keep using the traditional approach (not recommended due to retokenization):

```bash
bash train_deepswe_32b.sh
```

Note: Set `rllm.filter_token_mismatch=True` in the config to mask trajectories with token mismatches.

## Key Differences

| Feature | Traditional (`train_deepswe_32b.sh`) | SDK-based (`train_deepswe_32b_sdk.sh`) |
|---------|-------------------------------------|----------------------------------------|
| **Retokenization** | ⚠️ Prompts retokenized at each step | ✅ Zero retokenization |
| **Token IDs** | Mixed (prompts local, completions vLLM) | ✅ All from vLLM |
| **Storage** | In-memory episode_steps | ✅ SQLite traces |
| **Setup** | Direct training | Requires proxy server |
| **Token Fidelity** | ⚠️ Depends on tokenizer | ✅ 100% guaranteed |

## Architecture

### Traditional Flow
```
vLLM → VerlEngine → AgentExecutionEngine → Training
         ↓ (retokenize prompts)
       episode_steps
```

### SDK Flow
```
vLLM → LiteLLM Proxy → SQLite Trace → AgentSdkEngine → Training
         ↓ (capture token IDs)
     No retokenization!
```

## Configuration

### SDK-Specific Parameters

```yaml
sdk:
  proxy_url: "http://localhost:4000/v1"  # LiteLLM proxy URL
  session_name: "swe_agent_training"     # Trace session name
  groupby_key: "entry.instance_id"       # Group traces by task
```

### Removing AgentExecutionEngine Parameters

The following parameters are only for `AgentExecutionEngine` and can be removed when using SDK:

```yaml
# Remove these when using SDK:
rllm.env.name: swe
rllm.agent.name: sweagent
rllm.agent.max_steps: 50
rllm.agent.overlong_filter: True
rllm.agent.trajectory_timeout: 5400
rllm.mask_truncated_samples: False
```

## Troubleshooting

### "Connection refused" error
Make sure the LiteLLM proxy is running on port 4000:
```bash
ps aux | grep litellm
```

### "Dataset not found" error
Run the data preparation script:
```bash
python prepare_swe_data.py
```

### Token ID mismatch warnings (traditional approach only)
These warnings indicate retokenization issues. Consider migrating to the SDK approach.

## Performance

- **SDK overhead**: ~5-10% due to proxy and SQLite I/O
- **Benefit**: 100% token ID fidelity, better training stability
- **Recommended**: Use SDK for production training

## Additional Resources

- [Token ID Capture Flow Documentation](../../docs/architecture/token_id_capture_flow.md)
- [SDK Overview](../../docs/core-concepts/sdk.md)
- [AgentSdkEngine API](../../docs/api/engine/agent_sdk_engine.md)
