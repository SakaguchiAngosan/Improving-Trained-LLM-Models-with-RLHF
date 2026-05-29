# RLHF Platform Production Guide

This document describes the production-grade distributed RLHF platform built for multi-node GPU clusters. It is intended to complement the educational notebook material with a systems architecture view, deployment guidance, and operational best practices.

## Key Components

- `src/rlhf_platform/distributed/topology.py`
  - Defines an asymmetric model placement topology for Actor, Critic, Reference, and Reward models.
  - Supports FSDP + Tensor Parallelism for active models and inference replicas for frozen models.

- `src/rlhf_platform/distributed/comm_hooks.py`
  - Implements NCCL communication hooks to overlap gradient synchronization with backward computation.
  - Adds numerical safeguards for KL drift, gradient clipping, and NaN detection.

- `src/rlhf_platform/distributed/async_io.py`
  - Provides non-blocking checkpoint writing with a background writer thread.
  - Ensures only rank 0 performs disk I/O while all ranks can continue compute.

- `src/rlhf_platform/alignment/loss.py`
  - Aggregates PPO loss components with stable KL divergence, advantage computation, and value clipping.

- `src/rlhf_platform/alignment/ppo_engine.py`
  - Orchestrates the distributed PPO optimization step across models and ranks.

- `src/rlhf_platform/alignment/rollout.py`
  - Provides a rollout buffer and asynchronous generation pipeline to decouple sample collection from optimization.

- `src/rlhf_platform/utils/telemetry.py`
  - Tracks rank-aware structured telemetry events and NCCL profiling metrics.

## Deployment Configuration

Configuration files are stored under `configs/`:

- `configs/deepspeed_zero3.yaml`: ZeRO-3 optimizer configuration optimized for CPU offloading and overlapping communication.
- `configs/cluster_topology.yaml`: Logical cluster topology for a 16-GPU setup, including model placement, collective tuning, and training hyperparameters.

## Recommended Launch Pattern

1. Prepare the multi-node environment.
2. Launch `torchrun` with one process per GPU.
3. Use `configs/cluster_topology.yaml` to initialize distributed groups and model placement.
4. Start the training process and collect telemetry to disk.

Example:

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --master_addr="$MASTER_ADDR" \
  --master_port=29500 \
  train.py \
  --config configs/cluster_topology.yaml
```

## Operational Best Practices

- Enable NCCL debugging in staging and use `NCCL_DEBUG=INFO` only temporarily in production.
- Use bfloat16 for active model training and float32 for frozen inference replicas to preserve numerical stability.
- Keep a small number of asynchronous generator workers and a modest buffer capacity to avoid stale rollout data.
- Periodically flush checkpoints from the async writer and verify metadata consistency.

## Observability

The platform logs structured telemetry events with the following types:

- `ppo_step`
- `communication`
- `checkpoint`
- `memory`

Telemetry is intended to be consumed by downstream metrics systems such as Prometheus, OpenTelemetry, or log ingest pipelines.

## Notes

- This repo is designed for systems-level RLHF research and production prototyping.
- The current implementation is a framework skeleton; application-specific dataset handling, tokenizer integration, and training scheduler code should be built on top of the provided modules.
