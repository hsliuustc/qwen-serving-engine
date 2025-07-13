"""
Paged attention implementation for memory-efficient KV cache management.
Similar to vLLM's PagedAttention but simplified for educational purposes.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from loguru import logger
from config import PagedAttentionConfig

@dataclass
class BlockInfo:
    """Information about a memory block."""
    block_id: int
    sequence_id: str
    position: int  # Position in the sequence
    ref_count: int = 1
    last_used: float = 0.0

class BlockManager:
    """
    Manages memory blocks for paged attention.
    
    Each block stores KV cache for a fixed number of tokens.
    Blocks can be shared across sequences for prefix caching.
    """
    
    def __init__(self, config: PagedAttentionConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Block storage
        self.total_blocks = config.max_num_cached_blocks
        self.block_size = config.block_size
        self.free_blocks: List[int] = list(range(self.total_blocks))
        self.block_info: Dict[int, BlockInfo] = {}
        
        # Sequence to block mapping
        self.sequence_blocks: Dict[str, List[int]] = {}
        
        logger.info(f"Initialized BlockManager with {self.total_blocks} blocks of size {self.block_size}")
    
    def allocate_blocks(self, sequence_id: str, num_blocks: int) -> List[int]:
        """Allocate blocks for a sequence."""
        if len(self.free_blocks) < num_blocks:
            # Try to free some blocks using LRU
            self._free_lru_blocks(num_blocks - len(self.free_blocks))
        
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(f"Not enough free blocks. Requested: {num_blocks}, Available: {len(self.free_blocks)}")
        
        allocated_blocks = []
        for i in range(num_blocks):
            block_id = self.free_blocks.pop(0)
            self.block_info[block_id] = BlockInfo(
                block_id=block_id,
                sequence_id=sequence_id,
                position=i
            )
            allocated_blocks.append(block_id)
        
        self.sequence_blocks[sequence_id] = allocated_blocks
        logger.debug(f"Allocated {num_blocks} blocks for sequence {sequence_id}")
        return allocated_blocks
    
    def free_blocks(self, sequence_id: str):
        """Free blocks for a sequence."""
        if sequence_id not in self.sequence_blocks:
            return
        
        blocks = self.sequence_blocks[sequence_id]
        for block_id in blocks:
            if block_id in self.block_info:
                del self.block_info[block_id]
            self.free_blocks.append(block_id)
        
        del self.sequence_blocks[sequence_id]
        logger.debug(f"Freed {len(blocks)} blocks for sequence {sequence_id}")
    
    def get_blocks(self, sequence_id: str) -> List[int]:
        """Get blocks for a sequence."""
        return self.sequence_blocks.get(sequence_id, [])
    
    def _free_lru_blocks(self, num_blocks: int):
        """Free least recently used blocks."""
        # Sort blocks by last used time
        lru_blocks = sorted(
            self.block_info.items(),
            key=lambda x: x[1].last_used
        )
        
        freed = 0
        for block_id, block_info in lru_blocks:
            if freed >= num_blocks:
                break
            
            # Free the block
            sequence_id = block_info.sequence_id
            if sequence_id in self.sequence_blocks:
                self.sequence_blocks[sequence_id].remove(block_id)
                if not self.sequence_blocks[sequence_id]:
                    del self.sequence_blocks[sequence_id]
            
            del self.block_info[block_id]
            self.free_blocks.append(block_id)
            freed += 1
        
        logger.debug(f"Freed {freed} LRU blocks")

class PagedKVCache:
    """
    Key-Value cache using paged attention.
    
    Stores KV cache in blocks for memory efficiency and sharing.
    """
    
    def __init__(
        self,
        config: PagedAttentionConfig,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda"
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # Block manager
        self.block_manager = BlockManager(config, device)
        
        # KV cache storage: [num_layers, 2, total_blocks, block_size, num_heads, head_dim]
        # 2 for key and value
        self.kv_cache = torch.zeros(
            num_layers, 2, config.max_num_cached_blocks, config.block_size, num_heads, head_dim,
            dtype=getattr(torch, config.cache_dtype),
            device=device
        )
        
        logger.info(f"Initialized PagedKVCache with shape {self.kv_cache.shape}")
    
    def allocate_sequence(self, sequence_id: str, sequence_length: int) -> List[int]:
        """Allocate blocks for a new sequence."""
        num_blocks = math.ceil(sequence_length / self.config.block_size)
        if num_blocks > self.config.max_num_blocks_per_seq:
            raise ValueError(f"Sequence too long: {num_blocks} blocks > {self.config.max_num_blocks_per_seq}")
        
        return self.block_manager.allocate_blocks(sequence_id, num_blocks)
    
    def free_sequence(self, sequence_id: str):
        """Free blocks for a sequence."""
        self.block_manager.free_blocks(sequence_id)
    
    def store_kv(
        self,
        sequence_id: str,
        layer_idx: int,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        """Store key-value pair for a position in a sequence."""
        blocks = self.block_manager.get_blocks(sequence_id)
        if not blocks:
            raise ValueError(f"No blocks allocated for sequence {sequence_id}")
        
        block_idx = position // self.config.block_size
        position_in_block = position % self.config.block_size
        
        if block_idx >= len(blocks):
            raise ValueError(f"Position {position} exceeds allocated blocks for sequence {sequence_id}")
        
        block_id = blocks[block_idx]
        
        # Store key and value
        self.kv_cache[layer_idx, 0, block_id, position_in_block] = key
        self.kv_cache[layer_idx, 1, block_id, position_in_block] = value
        
        # Update last used time
        if block_id in self.block_manager.block_info:
            import time
            self.block_manager.block_info[block_id].last_used = time.time()
    
    def get_kv(self, sequence_id: str, layer_idx: int, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get key-value cache for a sequence up to max_length."""
        blocks = self.block_manager.get_blocks(sequence_id)
        if not blocks:
            # Return empty tensors if no blocks
            return (
                torch.empty(0, self.num_heads, self.head_dim, device=self.device),
                torch.empty(0, self.num_heads, self.head_dim, device=self.device)
            )
        
        keys = []
        values = []
        
        for block_idx, block_id in enumerate(blocks):
            start_pos = block_idx * self.config.block_size
            end_pos = min(start_pos + self.config.block_size, max_length)
            
            if start_pos >= max_length:
                break
            
            block_length = end_pos - start_pos
            
            # Get key and value from this block
            key_block = self.kv_cache[layer_idx, 0, block_id, :block_length]
            value_block = self.kv_cache[layer_idx, 1, block_id, :block_length]
            
            keys.append(key_block)
            values.append(value_block)
        
        if keys:
            return torch.cat(keys, dim=0), torch.cat(values, dim=0)
        else:
            return (
                torch.empty(0, self.num_heads, self.head_dim, device=self.device),
                torch.empty(0, self.num_heads, self.head_dim, device=self.device)
            )

class PagedAttention:
    """
    Paged attention mechanism for memory-efficient inference.
    
    Computes attention using paged KV cache for memory efficiency.
    """
    
    def __init__(self, config: PagedAttentionConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
    def compute_attention(
        self,
        query: torch.Tensor,  # [batch_size, seq_len, num_heads, head_dim]
        key_cache: torch.Tensor,  # [past_seq_len, num_heads, head_dim]
        value_cache: torch.Tensor,  # [past_seq_len, num_heads, head_dim]
        attention_mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention using paged KV cache.
        
        Args:
            query: Query tensor
            key_cache: Cached key tensor
            value_cache: Cached value tensor
            attention_mask: Optional attention mask
            scale: Optional scaling factor
            
        Returns:
            Attention output tensor
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        # Reshape query for attention computation
        query = query.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        if key_cache.numel() == 0:
            # No past key-value cache
            return torch.zeros(
                batch_size, num_heads, seq_len, head_dim,
                device=self.device,
                dtype=query.dtype
            ).transpose(1, 2)
        
        # Expand key and value cache for batch
        key_cache = key_cache.unsqueeze(0).expand(batch_size, -1, num_heads, head_dim)
        value_cache = value_cache.unsqueeze(0).expand(batch_size, -1, num_heads, head_dim)
        
        # Transpose for attention computation
        key_cache = key_cache.transpose(1, 2)  # [batch_size, num_heads, past_seq_len, head_dim]
        value_cache = value_cache.transpose(1, 2)  # [batch_size, num_heads, past_seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(query, key_cache.transpose(-2, -1)) * scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute attention output
        attention_output = torch.matmul(attention_weights, value_cache)
        
        # Transpose back to original format
        attention_output = attention_output.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        
        return attention_output
    
    def get_stats(self) -> Dict[str, Any]:
        """Get paged attention statistics."""
        return {
            "block_size": self.config.block_size,
            "max_blocks_per_seq": self.config.max_num_blocks_per_seq,
            "max_cached_blocks": self.config.max_num_cached_blocks,
            "cache_dtype": self.config.cache_dtype
        }
