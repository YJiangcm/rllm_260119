# rLLM Architecture Documentation

This directory contains detailed architectural documentation for the rLLM framework.

## Documents

### [Token ID Capture Flow](token_id_capture_flow.md)

Explains how rLLM captures token IDs directly from vLLM inference servers and uses them for training without any retokenization. This ensures 100% fidelity between inference and training tokens.

**Key Topics**:
- vLLM instrumentation mechanism for token ID capture
- Trace storage protocol and data structures
- Training pipeline integration (VERL and Tinker)
- Configuration and verification
- Benefits and implementation notes

**Read this if you want to understand**:
- How token IDs flow from vLLM to training
- Why rLLM doesn't retokenize text
- How to enable and verify token ID capture
- Integration points in the training pipeline

## Related Documentation

- [Core Concepts](../core-concepts/) - High-level framework concepts
- [SDK Documentation](../core-concepts/sdk.md) - Agent SDK overview
- [API Reference](../api/) - Detailed API documentation
