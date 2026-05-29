#!/usr/bin/env python3
"""Entrypoint for the RLHF distributed training platform."""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rlhf_platform.alignment.ppo_engine import DistributedPPOEngine
from rlhf_platform.alignment.rollout import AsyncRolloutPipeline, RolloutBuffer, RolloutGenerator
from rlhf_platform.distributed.async_io import DistributedCheckpointManager
from rlhf_platform.distributed.comm_hooks import CommunicationHooks
from rlhf_platform.distributed.topology import load_topology_from_yaml
from rlhf_platform.utils.telemetry import RankAwareTelemetry


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rlhf_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLHF distributed training entrypoint.")
    parser.add_argument("--config", default="configs/cluster_topology.yaml", help="Path to cluster topology YAML config.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for async checkpoints.")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of PPO update steps to execute.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for rollout generation.")
    parser.add_argument("--max-response-length", type=int, default=64, help="Max generated token length for each rollout.")
    parser.add_argument("--prompts", nargs="*", default=None, help="Optional list of prompts to use for generation.")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU execution rather than GPU/distributed.")
    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() and not args.use_cpu else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        logger.info(f"Initialized distributed process group: rank={rank}, world_size={world_size}, backend={backend}")
    else:
        logger.info("Running in single-process mode.")

    return world_size, rank, local_rank


def create_query_sampler(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int,
    device: torch.device,
    max_prompt_length: int = 64,
) -> callable:
    if not prompts:
        prompts = [
            "Summarize the following paragraph.",
            "Explain the intent of the user request.",
            "Write a short answer describing why RLHF improves alignment.",
        ]

    def _pad_to_batch(encodings: List[torch.Tensor]) -> torch.Tensor:
        max_length = max(t.shape[0] for t in encodings)
        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        batch = torch.full((len(encodings), max_length), pad_token, dtype=torch.long, device=device)
        for i, tensor in enumerate(encodings):
            batch[i, : tensor.shape[0]] = tensor
        return batch

    def sampler() -> torch.Tensor:
        encodings = []
        for _ in range(batch_size):
            prompt = random.choice(prompts)
            encoding = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_length,
            )
            encodings.append(encoding.input_ids[0])
        return _pad_to_batch(encodings)

    return sampler


def build_models(
    actor_name: str,
    critic_name: str,
    reference_name: str,
    reward_name: str,
    device: torch.device,
    use_fp16: bool = True,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module, AutoTokenizer]:
    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(actor_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor_model = AutoModelForCausalLM.from_pretrained(actor_name).to(device, dtype=dtype)
    reference_model = AutoModelForCausalLM.from_pretrained(reference_name).to(device, dtype=torch.float32)
    critic_model = AutoModelForSequenceClassification.from_pretrained(
        critic_name,
        num_labels=1,
        problem_type="regression",
    ).to(device, dtype=torch.float32)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_name,
        num_labels=1,
        problem_type="regression",
    ).to(device, dtype=torch.float32)

    reference_model.eval()
    reward_model.eval()
    for model in [reference_model, reward_model]:
        for param in model.parameters():
            param.requires_grad = False

    return actor_model, critic_model, reference_model, reward_model, tokenizer


def wrap_distributed(
    model: torch.nn.Module,
    process_group,
    rank: int,
    local_rank: int,
    use_cuda: bool,
) -> torch.nn.Module:
    if process_group is None:
        return model

    if use_cuda:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            process_group=process_group,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=None,
            process_group=process_group,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    logger.info(f"Wrapped model {model.__class__.__name__} in DistributedDataParallel on rank {rank}.")
    return ddp_model


def get_state_dict(model: torch.nn.Module) -> dict:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def main() -> None:
    global args
    args = parse_args()

    world_size, rank, local_rank = init_distributed(args)
    use_cuda = torch.cuda.is_available() and not args.use_cpu
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    topology = load_topology_from_yaml(args.config, rank=rank, local_rank=local_rank)
    if not dist.is_initialized():
        logger.warning("Distributed groups are not initialized; topology groups will not be created.")

    actor_model, critic_model, reference_model, reward_model, tokenizer = build_models(
        actor_name=topology.actor_placement.model_name,
        critic_name=topology.critic_placement.model_name,
        reference_name=topology.reference_placement.model_name,
        reward_name=topology.reward_placement.model_name,
        device=device,
        use_fp16=True,
    )

    if dist.is_initialized() and topology.is_active_rank():
        actor_model = wrap_distributed(actor_model, topology.actor_group, rank, local_rank, use_cuda)
        critic_model = wrap_distributed(critic_model, topology.critic_group, rank, local_rank, use_cuda)
        CommunicationHooks.register_overlap_hook(actor_model)
        CommunicationHooks.register_gradient_clipping_hook(actor_model)
        CommunicationHooks.register_nan_check_hook(actor_model)

    optimizer = AdamW(
        list(actor_model.parameters()) + list(critic_model.parameters()),
        lr=1e-5,
        eps=1e-8,
    )

    sampler = create_query_sampler(tokenizer, args.prompts or [], args.batch_size, device)
    buffer = RolloutBuffer(capacity=512, device=device.type)
    generator = RolloutGenerator(
        policy_model=actor_model.module if hasattr(actor_model, "module") else actor_model,
        reference_model=reference_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        max_response_length=args.max_response_length,
    )
    pipeline = AsyncRolloutPipeline(
        generator=generator,
        buffer=buffer,
        query_sampler=sampler,
        batch_size=args.batch_size,
        num_generator_threads=2,
    )
    pipeline.start_generators()

    ppo_engine = DistributedPPOEngine(
        actor_model=actor_model,
        critic_model=critic_model,
        reference_model=reference_model,
        reward_model=reward_model,
        optimizer=optimizer,
        world_size=world_size,
        rank=rank,
        gradient_accumulation_steps=8,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = DistributedCheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        world_size=world_size,
        rank=rank,
        save_frequency=100,
    )

    telemetry = RankAwareTelemetry(rank=rank, world_size=world_size, log_file=None)

    try:
        while ppo_engine.step < args.num_steps:
            if buffer.size() < max(1, args.batch_size):
                time.sleep(0.2)
                continue

            start_ts = time.time()
            metrics = ppo_engine.ppo_step()
            duration_ms = (time.time() - start_ts) * 1000.0

            if metrics:
                telemetry.log_ppo_step(
                    step=ppo_engine.step,
                    actor_loss=metrics.get("actor_loss", 0.0),
                    value_loss=metrics.get("value_loss", 0.0),
                    kl_divergence=metrics.get("kl_divergence", 0.0),
                    policy_entropy=metrics.get("policy_entropy", 0.0),
                    duration_ms=duration_ms,
                )
                if rank == 0 and ppo_engine.step % 50 == 0:
                    logger.info(f"Step={ppo_engine.step} metrics={metrics}")

            if rank == 0 and ppo_engine.step % 100 == 0 and ppo_engine.step > 0:
                checkpoint_manager.save_checkpoint(
                    step=ppo_engine.step,
                    model_name=topology.actor_placement.model_name,
                    model_state_dict=get_state_dict(actor_model),
                    optimizer_state_dict=optimizer.state_dict(),
                    metrics=metrics,
                )

            if rank == 0 and ppo_engine.step and ppo_engine.step % 10 == 0:
                allocated = torch.cuda.memory_allocated(device) / 1024.0 / 1024.0 if use_cuda else 0.0
                reserved = torch.cuda.memory_reserved(device) / 1024.0 / 1024.0 if use_cuda else 0.0
                max_allocated = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0 if use_cuda else 0.0
                telemetry.log_memory_usage(
                    allocated_mb=allocated,
                    reserved_mb=reserved,
                    max_allocated_mb=max_allocated,
                )
    except KeyboardInterrupt:
        logger.info("Interrupted by user; attempting graceful shutdown.")
    finally:
        pipeline.stop_generators()
        checkpoint_manager.flush_and_shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
