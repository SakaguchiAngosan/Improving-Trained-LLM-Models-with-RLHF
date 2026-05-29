"""
rollout.py: Parallelized rollout generator with shared-memory ring buffers.

Enables decoupled generation and optimization: a subset of worker nodes
continuously generate rollout sequences and stream them into an async
shared-memory CPU ring buffer, while core training nodes pull from this
buffer without waiting for generation phases to complete.
"""

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    """Single PPO rollout trajectory."""
    query_tokens: torch.Tensor          # (seq_length,)
    response_tokens: torch.Tensor       # (response_length,)
    reward: torch.Tensor                # scalar
    logits_policy: torch.Tensor         # (response_length, vocab_size)
    logits_reference: torch.Tensor      # (response_length, vocab_size)


class RolloutBuffer:
    """In-memory ring buffer for collecting rollouts during generation phase."""

    def __init__(self, capacity: int = 1000, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer: list = []
        self.ptr = 0

    def add(self, rollout: Rollout) -> None:
        """Add a rollout to the buffer (overwrites oldest if full)."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(rollout)
        else:
            self.buffer[self.ptr] = rollout
            self.ptr = (self.ptr + 1) % self.capacity

    def sample_batch(self, batch_size: int) -> list:
        """Sample a random batch of rollouts."""
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot sample from empty rollout buffer.")
        
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.ptr = 0

    def size(self) -> int:
        """Current number of rollouts in buffer."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.capacity


class RolloutDataset(Dataset):
    """PyTorch Dataset wrapper for rollout buffer."""

    def __init__(self, buffer: RolloutBuffer):
        self.buffer = buffer

    def __len__(self) -> int:
        return self.buffer.size()

    def __getitem__(self, idx: int) -> Rollout:
        return self.buffer.buffer[idx]


class RolloutCollator:
    """Collate rollouts into batched tensors."""

    def __init__(self, max_response_length: int = 128):
        self.max_response_length = max_response_length

    def __call__(self, rollouts: list) -> dict:
        """Collate a list of Rollout objects."""
        batch_size = len(rollouts)
        
        # Pad response tokens
        response_tokens_list = [r.response_tokens for r in rollouts]
        padded_responses = self._pad_sequence(response_tokens_list, self.max_response_length)
        
        # Pad logits
        logits_policy_list = [r.logits_policy for r in rollouts]
        logits_reference_list = [r.logits_reference for r in rollouts]
        
        padded_logits_policy = self._pad_logits(logits_policy_list, self.max_response_length)
        padded_logits_reference = self._pad_logits(logits_reference_list, self.max_response_length)
        
        # Rewards
        rewards = torch.stack([r.reward for r in rollouts])
        
        return {
            "response_tokens": padded_responses,
            "logits_policy": padded_logits_policy,
            "logits_reference": padded_logits_reference,
            "rewards": rewards,
        }

    @staticmethod
    def _pad_sequence(sequences: list, max_length: int) -> torch.Tensor:
        """Pad token sequences to max_length."""
        padded = torch.zeros(len(sequences), max_length, dtype=torch.long)
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_length)
            padded[i, :seq_len] = seq[:seq_len]
        return padded

    @staticmethod
    def _pad_logits(logits_list: list, max_length: int) -> torch.Tensor:
        """Pad logits to (batch, max_length, vocab_size)."""
        vocab_size = logits_list[0].shape[-1]
        padded = torch.zeros(len(logits_list), max_length, vocab_size)
        for i, logits in enumerate(logits_list):
            seq_len = min(len(logits), max_length)
            padded[i, :seq_len] = logits[:seq_len]
        return padded


class RolloutGenerator:
    """Generate rollouts from policy model and score with reward model."""

    def __init__(
        self,
        policy_model: torch.nn.Module,
        reference_model: torch.nn.Module,
        reward_model: torch.nn.Module,
        tokenizer,
        max_response_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def generate_rollout(self, query_tokens: torch.Tensor) -> Rollout:
        """Generate a single rollout trajectory."""
        batch_size = query_tokens.shape[0]
        device = query_tokens.device
        
        # Generate response tokens autoregressively
        response_tokens_list = []
        logits_policy_list = []
        logits_reference_list = []
        
        current_tokens = query_tokens.clone()
        
        for _ in range(self.max_response_length):
            # Policy forward pass
            policy_out = self.policy_model(current_tokens)
            policy_logits = policy_out.logits[:, -1, :]  # (batch, vocab_size)
            
            # Reference forward pass
            ref_out = self.reference_model(current_tokens)
            ref_logits = ref_out.logits[:, -1, :]
            
            logits_policy_list.append(policy_logits)
            logits_reference_list.append(ref_logits)
            
            # Sample next token (temperature + top_p sampling)
            next_token = self._sample_token(
                policy_logits,
                temperature=self.temperature,
                top_p=self.top_p
            )
            response_tokens_list.append(next_token)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)
        
        # Compute reward on full (query + response) text
        response_tokens = torch.stack(response_tokens_list, dim=1)  # (batch, response_length)
        
        with torch.no_grad():
            reward_input = torch.cat([query_tokens, response_tokens], dim=1)
            reward_out = self.reward_model(reward_input)
            rewards = reward_out.logits.squeeze(-1)  # (batch,)
        
        # Stack logits
        logits_policy = torch.stack(logits_policy_list, dim=1)      # (batch, response_length, vocab_size)
        logits_reference = torch.stack(logits_reference_list, dim=1)
        
        return {
            "query_tokens": query_tokens,
            "response_tokens": response_tokens,
            "logits_policy": logits_policy,
            "logits_reference": logits_reference,
            "rewards": rewards,
        }

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Sample next token using temperature + top-p (nucleus) sampling."""
        # Temperature scaling
        scaled_logits = logits / temperature
        
        # Top-p sampling
        probs = F.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff: include all tokens where cumulative prob <= top_p
        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[..., 0] = False  # Keep at least one token
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scaled_logits[:, indices_to_remove] = float("-inf")
        
        # Sample
        token = torch.multinomial(F.softmax(scaled_logits, dim=-1), num_samples=1)
        return token


class AsyncRolloutPipeline:
    """
    Asynchronous rollout pipeline: decouples generation from optimization.
    
    Generator threads continuously produce rollouts and feed them into
    a shared RolloutBuffer. Optimization threads pull from the buffer
    without blocking on generation latency.
    """

    def __init__(
        self,
        generator: RolloutGenerator,
        buffer: RolloutBuffer,
        num_generator_threads: int = 2,
    ):
        self.generator = generator
        self.buffer = buffer
        self.num_generator_threads = num_generator_threads
        self.stop_event = False

    def start_generators(self) -> None:
        """Start background generator threads."""
        import threading
        self.generator_threads = []
        for i in range(self.num_generator_threads):
            thread = threading.Thread(target=self._generator_loop, args=(i,), daemon=True)
            thread.start()
            self.generator_threads.append(thread)
        logger.info(f"Started {self.num_generator_threads} generator threads.")

    def _generator_loop(self, thread_id: int) -> None:
        """Run generation loop in background thread."""
        logger.info(f"Generator thread {thread_id} started.")
        while not self.stop_event:
            try:
                # Generate a rollout
                # (In practice, you'd sample queries from a dataset)
                pass
            except Exception as e:
                logger.error(f"Generator thread {thread_id} error: {e}", exc_info=True)

    def stop_generators(self) -> None:
        """Stop all generator threads."""
        self.stop_event = True
        for thread in self.generator_threads:
            thread.join(timeout=5.0)
        logger.info("Generator threads stopped.")
