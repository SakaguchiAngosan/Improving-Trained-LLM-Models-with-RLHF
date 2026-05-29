"""
topology.py: Asymmetric multi-model placement grid for distributed RLHF.

Elite teams at OpenAI, Anthropic, and DeepMind recognize that running Actor, Reference,
Critic, and Reward models with identical distributed strategies creates massive
synchronization bottlenecks. This module implements a hybrid topology:

- Actor & Critic (Active): ZeRO-3 + Tensor Parallelism across nodes
- Reference & Reward (Frozen): Shared inference subnet with replica placement
  and non-gradient tracking, reducing memory footprints and communication overhead.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


class ModelRole(Enum):
    """Role of a model in the RLHF pipeline."""
    ACTOR = "actor"              # Policy being trained
    CRITIC = "critic"             # Value network being trained
    REFERENCE = "reference"       # Frozen baseline (for KL compute)
    REWARD = "reward"             # Frozen reward scorer


class ParallelStrategy(Enum):
    """Distributed execution strategy."""
    FSDP_TP = "fsdp_tp"           # FSDP + Tensor Parallelism (active models)
    INFERENCE = "inference"       # Replica placement for frozen models
    STANDALONE = "standalone"     # Single-GPU or DDP


@dataclass
class ModelPlacement:
    """Placement strategy for a single model within the cluster."""
    role: ModelRole
    model_name: str
    parameter_count: int
    strategy: ParallelStrategy
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    device_list: List[int] = field(default_factory=list)
    precision: str = "fp16"        # fp16, bf16, int8
    offload_to_cpu: bool = False

    def __post_init__(self):
        if self.strategy == ParallelStrategy.FSDP_TP:
            assert self.tensor_parallel_size >= 1, "Tensor parallelism must be >= 1"
            assert self.data_parallel_size >= 1, "Data parallelism must be >= 1"
        if self.role in [ModelRole.REFERENCE, ModelRole.REWARD]:
            assert self.strategy != ParallelStrategy.FSDP_TP, \
                "Frozen models should use INFERENCE or STANDALONE strategy"


@dataclass
class ClusterTopology:
    """
    Defines the complete asymmetric topology for a multi-node RLHF cluster.
    Maps Actor, Critic, Reference, and Reward models to separate device grids.
    """
    world_size: int
    rank: int
    local_rank: int
    num_nodes: int
    gpus_per_node: int
    
    actor_placement: Optional[ModelPlacement] = None
    critic_placement: Optional[ModelPlacement] = None
    reference_placement: Optional[ModelPlacement] = None
    reward_placement: Optional[ModelPlacement] = None
    
    # Device group mappings for collective communication
    actor_group: Optional[dist.ProcessGroup] = None
    critic_group: Optional[dist.ProcessGroup] = None
    reference_group: Optional[dist.ProcessGroup] = None
    reward_group: Optional[dist.ProcessGroup] = None

    def validate(self):
        """Ensure no rank is assigned to multiple active model groups."""
        active_groups = []
        for placement, group in [
            (self.actor_placement, self.actor_group),
            (self.critic_placement, self.critic_group),
        ]:
            if placement is not None and placement.strategy == ParallelStrategy.FSDP_TP:
                if self.rank in placement.device_list:
                    active_groups.append(placement.role)
        
        assert len(active_groups) <= 1, \
            f"Rank {self.rank} assigned to multiple active model groups: {active_groups}. " \
            "Each rank must focus on at most one active (training) model."

    def to_dict(self) -> Dict:
        """Serialize topology to dictionary (for logging/verification)."""
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "actor": self.actor_placement.__dict__ if self.actor_placement else None,
            "critic": self.critic_placement.__dict__ if self.critic_placement else None,
            "reference": self.reference_placement.__dict__ if self.reference_placement else None,
            "reward": self.reward_placement.__dict__ if self.reward_placement else None,
        }

    def to_json(self) -> str:
        """Export topology as JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @property
    def device(self) -> torch.device:
        """Get the primary device for this rank."""
        return torch.device(f"cuda:{self.local_rank}")


class TopologyBuilder:
    """Builder pattern for constructing asymmetric cluster topologies."""

    def __init__(self, world_size: int, rank: int, gpus_per_node: int):
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.local_rank = rank % gpus_per_node
        self.num_nodes = world_size // gpus_per_node
        self.topology = ClusterTopology(
            world_size=world_size,
            rank=rank,
            local_rank=self.local_rank,
            num_nodes=self.num_nodes,
            gpus_per_node=gpus_per_node,
        )

    def add_actor(
        self,
        model_name: str,
        param_count: int,
        tensor_parallel_size: int = 1,
        data_parallel_size: Optional[int] = None,
        precision: str = "fp16",
    ) -> "TopologyBuilder":
        """Add Actor (policy) model with FSDP + TP strategy."""
        if data_parallel_size is None:
            data_parallel_size = self.world_size // tensor_parallel_size
        
        device_list = list(range(self.world_size))
        self.topology.actor_placement = ModelPlacement(
            role=ModelRole.ACTOR,
            model_name=model_name,
            parameter_count=param_count,
            strategy=ParallelStrategy.FSDP_TP,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            device_list=device_list,
            precision=precision,
        )
        return self

    def add_critic(
        self,
        model_name: str,
        param_count: int,
        tensor_parallel_size: int = 1,
        data_parallel_size: Optional[int] = None,
        precision: str = "fp16",
    ) -> "TopologyBuilder":
        """Add Critic (value network) with FSDP + TP strategy."""
        if data_parallel_size is None:
            data_parallel_size = self.world_size // tensor_parallel_size
        
        device_list = list(range(self.world_size))
        self.topology.critic_placement = ModelPlacement(
            role=ModelRole.CRITIC,
            model_name=model_name,
            parameter_count=param_count,
            strategy=ParallelStrategy.FSDP_TP,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            device_list=device_list,
            precision=precision,
        )
        return self

    def add_reference(
        self,
        model_name: str,
        param_count: int,
        replica_count: int = 1,
        precision: str = "fp16",
    ) -> "TopologyBuilder":
        """Add Reference model (frozen, inference-only) with replica placement."""
        device_list = list(range(min(replica_count * self.gpus_per_node, self.world_size)))
        self.topology.reference_placement = ModelPlacement(
            role=ModelRole.REFERENCE,
            model_name=model_name,
            parameter_count=param_count,
            strategy=ParallelStrategy.INFERENCE,
            device_list=device_list,
            precision=precision,
        )
        return self

    def add_reward(
        self,
        model_name: str,
        param_count: int,
        replica_count: int = 1,
        precision: str = "fp16",
    ) -> "TopologyBuilder":
        """Add Reward model (frozen, inference-only) with replica placement."""
        device_list = list(range(min(replica_count * self.gpus_per_node, self.world_size)))
        self.topology.reward_placement = ModelPlacement(
            role=ModelRole.REWARD,
            model_name=model_name,
            parameter_count=param_count,
            strategy=ParallelStrategy.INFERENCE,
            device_list=device_list,
            precision=precision,
        )
        return self

    def build(self) -> ClusterTopology:
        """Build and validate the topology."""
        self.topology.validate()
        return self.topology


def create_default_topology(
    world_size: int,
    rank: int,
    gpus_per_node: int,
    actor_model: str = "facebook/opt-1.3b",
    critic_model: str = "facebook/opt-1.3b",
    reference_model: str = "facebook/opt-1.3b",
    reward_model: str = "microsoft/deberta-v3-base",
) -> ClusterTopology:
    """
    Create a default asymmetric topology suitable for a 16-GPU cluster
    with 2 nodes of 8 GPUs each.
    
    Configuration:
    - Actor & Critic: FSDP with 2-way Tensor Parallelism (4 data-parallel groups)
    - Reference: Inference replica on first 2 GPUs
    - Reward: Inference replica on GPUs 2-4
    """
    builder = TopologyBuilder(world_size, rank, gpus_per_node)
    
    builder.add_actor(
        model_name=actor_model,
        param_count=1_300_000_000,
        tensor_parallel_size=2,
        data_parallel_size=world_size // 2,
        precision="fp16",
    )
    
    builder.add_critic(
        model_name=critic_model,
        param_count=1_300_000_000,
        tensor_parallel_size=2,
        data_parallel_size=world_size // 2,
        precision="fp16",
    )
    
    builder.add_reference(
        model_name=reference_model,
        param_count=1_300_000_000,
        replica_count=1,
        precision="fp16",
    )
    
    builder.add_reward(
        model_name=reward_model,
        param_count=300_000_000,
        replica_count=1,
        precision="fp16",
    )
    
    return builder.build()
