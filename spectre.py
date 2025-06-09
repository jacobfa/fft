"""
SPECTRE: Spectral Token Routing - An FFT-Based Efficient Drop-In Replacement to Self-Attention

This module implements the SPECTRE algorithm as described in the paper:
"SPECTRE: An FFT-Based Efficient Drop-In Replacement to Self-Attention for Long Contexts"
by Jacob Fein-Ashley, Neelesh Gupta, Rajgopal Kannan, and Viktor Prasanna (2025)

Key components:
- SPECTRELayer: Core frequency-mixing layer with O(n log n) complexity
- PrefixFFTCache: Efficient caching mechanism for autoregressive generation
- SPECTREBlock: Complete transformer block with SPECTRE layer
- Optional WaveletRefinementModule: For capturing local details
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np


class ComplexModReLU(nn.Module):
    """
    Modified ReLU for complex numbers that applies ReLU to magnitude while preserving phase.
    
    For a complex number z = r * e^(i*θ) where r = |z| and θ = arg(z):
    modReLU(z) = (r + b) * e^(i*θ) if r + b > 0, else 0
    """
    def __init__(self, bias: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply modReLU to complex tensor."""
        magnitude = torch.abs(z)
        phase = torch.angle(z)
        
        # Apply ReLU-like threshold to magnitude
        new_magnitude = F.relu(magnitude + self.bias)
        
        # Reconstruct complex number with new magnitude and same phase
        return new_magnitude * torch.exp(1j * phase)


class PrefixFFTCache:
    """
    Maintains running real-FFT coefficients and mean queries during generation.
    Enables constant-time updates for autoregressive decoding.
    """
    def __init__(self, max_seq_len: int, d_model: int, n_heads: int, device: torch.device):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.d_head = d_model // n_heads
        
        # Number of unique FFT coefficients for real signals
        self.n_freq = max_seq_len // 2 + 1
        
        # Initialize cache tensors
        self.prefix_fft = torch.zeros(
            n_heads, self.n_freq, self.d_head, 
            dtype=torch.complex64, device=device
        )
        
        # Ring buffers for values and queries
        self.V_buf = torch.zeros(n_heads, max_seq_len, self.d_head, device=device)
        self.Q_buf = torch.zeros(n_heads, max_seq_len, self.d_head, device=device)
        
        # Running sum of queries for global context
        self.sum_q = torch.zeros(n_heads, self.d_head, device=device)
        
        # Pre-compute twiddle factors for efficiency
        self._precompute_twiddle_factors()
        
        # Track current position in ring buffer
        self.position = 0
        self.filled_len = 0
    
    def _precompute_twiddle_factors(self):
        """Pre-compute e^(-j*2π*k*t/N) for all k and t."""
        k = torch.arange(self.n_freq, device=self.device).unsqueeze(1)
        t = torch.arange(self.max_seq_len, device=self.device).unsqueeze(0)
        self.twiddle = torch.exp(-2j * math.pi * k * t / self.max_seq_len)
    
    def prefill(self, V: torch.Tensor, Q: torch.Tensor) -> None:
        """
        One-shot initialization for the prompt.
        
        Args:
            V: Values tensor [batch, heads, seq_len, d_head]
            Q: Queries tensor [batch, heads, seq_len, d_head]
        """
        batch, n_heads, seq_len, d_head = V.shape
        assert batch == 1, "Batch size must be 1 for caching"
        
        # Pad to max sequence length
        V_padded = F.pad(V[0], (0, 0, 0, self.max_seq_len - seq_len))
        
        # Compute FFT and store non-redundant coefficients
        self.prefix_fft = torch.fft.rfft(V_padded, dim=1, norm='ortho')
        
        # Initialize ring buffers
        self.V_buf[:, :seq_len] = V[0]
        self.Q_buf[:, :seq_len] = Q[0]
        
        # Initialize running sum
        self.sum_q = Q[0].sum(dim=1)
        
        self.filled_len = seq_len
        self.position = seq_len % self.max_seq_len
    
    def decode_step(self, v_t: torch.Tensor, q_t: torch.Tensor, t: int) -> None:
        """
        Incremental update for a single new token.
        
        Args:
            v_t: New value token [batch=1, heads, 1, d_head]
            q_t: New query token [batch=1, heads, 1, d_head]
            t: Current time step
        """
        v_t = v_t[0, :, 0]  # [heads, d_head]
        q_t = q_t[0, :, 0]  # [heads, d_head]
        
        # Get old values if we're wrapping around
        if t >= self.max_seq_len:
            v_old = self.V_buf[:, self.position]
            q_old = self.Q_buf[:, self.position]
        else:
            v_old = torch.zeros_like(v_t)
            q_old = torch.zeros_like(q_t)
        
        # Update FFT coefficients
        for k in range(self.n_freq):
            # Remove old contribution
            if t >= self.max_seq_len:
                old_contrib = v_old * self.twiddle[k, (t - self.max_seq_len) % self.max_seq_len]
                self.prefix_fft[:, k] -= old_contrib
            
            # Add new contribution
            new_contrib = v_t * self.twiddle[k, t % self.max_seq_len]
            self.prefix_fft[:, k] += new_contrib
        
        # Update ring buffers
        self.V_buf[:, self.position] = v_t
        self.Q_buf[:, self.position] = q_t
        
        # Update running sum
        if t >= self.max_seq_len:
            self.sum_q = self.sum_q - q_old + q_t
        else:
            self.sum_q = self.sum_q + q_t
        
        # Update position
        self.position = (self.position + 1) % self.max_seq_len
        self.filled_len = min(self.filled_len + 1, self.max_seq_len)
    
    def get_current_fft(self) -> torch.Tensor:
        """Get current FFT coefficients."""
        return self.prefix_fft
    
    def get_mean_query(self) -> torch.Tensor:
        """Get normalized mean query for spectral gating."""
        return self.sum_q / self.filled_len


class SpectralGate(nn.Module):
    """
    Generates content-adaptive spectral gates based on global context.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, 
                 share_gates: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_freq = max_seq_len // 2 + 1
        self.share_gates = share_gates
        
        # Two-layer MLP to generate complex gates
        hidden_dim = d_model if share_gates else d_model * 2
        self.mlp = nn.Sequential(
            nn.Linear(self.d_head, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_freq * 2)  # Real and imaginary parts
        )
        
        # Layer norm for mean query
        self.norm = nn.LayerNorm(self.d_head)
    
    def forward(self, mean_query: torch.Tensor) -> torch.Tensor:
        """
        Generate spectral gates from mean query.
        
        Args:
            mean_query: [batch, heads, d_head] or [heads, d_head]
            
        Returns:
            Complex gates [batch, heads, n_freq] or [heads, n_freq]
        """
        # Normalize mean query
        q_norm = self.norm(mean_query)
        
        # Generate gates via MLP
        if self.share_gates:
            # Use mean across heads
            if q_norm.dim() == 3:
                q_shared = q_norm.mean(dim=1)  # [batch, d_head]
            else:
                q_shared = q_norm.mean(dim=0)  # [d_head]
            gates_real_imag = self.mlp(q_shared)  # [batch, n_freq*2] or [n_freq*2]
        else:
            gates_real_imag = self.mlp(q_norm)  # [batch, heads, n_freq*2] or [heads, n_freq*2]
        
        # Split into real and imaginary parts
        if gates_real_imag.dim() == 1:
            real = gates_real_imag[:self.n_freq]
            imag = gates_real_imag[self.n_freq:]
            gates = torch.complex(real, imag)
        elif gates_real_imag.dim() == 2:
            real = gates_real_imag[..., :self.n_freq]
            imag = gates_real_imag[..., self.n_freq:]
            gates = torch.complex(real, imag)
            if self.share_gates and mean_query.dim() == 3:
                gates = gates.unsqueeze(1).expand(-1, self.n_heads, -1)
        else:
            real = gates_real_imag[..., :self.n_freq]
            imag = gates_real_imag[..., self.n_freq:]
            gates = torch.complex(real, imag)
        
        return gates


class SPECTRELayer(nn.Module):
    """
    SPECTRE: Frequency-domain token mixer with optional low-rank outer product
    and wavelet refinement branch. Drop-in replacement for Multi-Head Attention.
    
    Complexity: O(n log n) instead of O(n^2) for self-attention.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 8192,
        low_rank: Optional[int] = None,
        use_wavelet: bool = False,
        share_gates: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.low_rank = low_rank
        self.use_wavelet = use_wavelet
        
        # Query and Value projections (per head)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Spectral gating module
        self.spectral_gate = SpectralGate(d_model, n_heads, max_seq_len, share_gates)
        
        # Optional low-rank Toeplitz convolution
        if low_rank is not None:
            self.toeplitz_kernel = nn.Parameter(
                torch.randn(n_heads, 2 * low_rank + 1, dtype=torch.complex64) * 0.02
            )
        
        # Complex activation
        self.modrelu = ComplexModReLU(bias=0.1)
        
        # Optional wavelet refinement module
        if use_wavelet:
            self.wrm = WaveletRefinementModule(d_model, n_heads)
            self.skip_controller = nn.Linear(self.d_head, 1)
            self.skip_init = 0.9  # Initially skip 90% of the time
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        cache: Optional[PrefixFFTCache] = None,
        incremental: bool = False,
        position: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[PrefixFFTCache]]:
        """
        Forward pass of SPECTRE layer.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            cache: Optional prefix-FFT cache for incremental decoding
            incremental: Whether this is an incremental decoding step
            position: Current position for positional phase injection
            
        Returns:
            Output tensor and updated cache
        """
        batch, seq_len, _ = x.shape
        
        # Project to queries and values
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        if incremental and cache is not None:
            # Incremental decoding path
            assert seq_len == 1 and position is not None
            
            # Update cache with new token
            cache.decode_step(V, Q, position)
            
            # Get current FFT coefficients and mean query
            V_fft = cache.get_current_fft()
            mean_q = cache.get_mean_query()
            
            # Generate spectral gates
            gates = self.spectral_gate(mean_q)  # [heads, n_freq]
            
            # Get actual number of frequency bins
            actual_n_freq = V_fft.shape[1]
            
            # Truncate or pad gates to match actual frequency bins
            if gates.shape[-1] > actual_n_freq:
                gates = gates[..., :actual_n_freq]
            elif gates.shape[-1] < actual_n_freq:
                gates = F.pad(gates, (0, actual_n_freq - gates.shape[-1]))
            
            # Reshape gates to match V_fft dimensions
            gates = gates.unsqueeze(-1)  # [heads, n_freq, 1]
            
            # Apply positional phase for current position
            k = torch.arange(actual_n_freq, device=x.device)
            pos_phase = torch.exp(2j * math.pi * k * position / cache.max_seq_len)
            gates = gates * pos_phase.unsqueeze(0).unsqueeze(-1)
            
            # Apply gates and activation
            V_fft_gated = gates * V_fft
            if self.low_rank is not None:
                V_fft_gated = self._apply_toeplitz(V_fft_gated)
            V_fft_gated = self.modrelu(V_fft_gated)
            
            # Inverse FFT to get output
            V_mixed = torch.fft.irfft(V_fft_gated, n=cache.max_seq_len, dim=1, norm='ortho')
            
            # Extract only the last token for incremental decoding
            # V_mixed shape: [heads, max_seq_len, d_head]
            # We want the token at the current position
            current_pos = (cache.position - 1) % cache.max_seq_len
            V_mixed = V_mixed[:, current_pos:current_pos+1]  # [heads, 1, d_head]
            
            # Add batch dimension
            V_mixed = V_mixed.unsqueeze(0)  # [1, heads, 1, d_head]
            
        else:
            # Full sequence processing
            if cache is not None and not incremental:
                # Pre-fill the cache
                cache.prefill(V, Q)
            
            # Compute mean query for gating
            mean_q = Q.mean(dim=2)  # [batch, heads, d_head]
            
            # Apply FFT to values
            V_fft = torch.fft.rfft(V, dim=2, norm='ortho')
            
            # Generate and apply spectral gates
            gates = self.spectral_gate(mean_q)  # [batch, heads, n_freq]
            
            # Get actual number of frequency bins from V_fft
            actual_n_freq = V_fft.shape[2]
            
            # Truncate or pad gates to match actual frequency bins
            if gates.shape[-1] > actual_n_freq:
                gates = gates[..., :actual_n_freq]
            elif gates.shape[-1] < actual_n_freq:
                gates = F.pad(gates, (0, actual_n_freq - gates.shape[-1]))
            
            # Reshape gates to match V_fft dimensions
            gates = gates.unsqueeze(-1)  # [batch, heads, n_freq, 1]
            
            # Apply positional phase if needed
            if position is not None:
                k = torch.arange(actual_n_freq, device=x.device)
                pos_phase = torch.exp(2j * math.pi * k * position / seq_len)
                gates = gates * pos_phase.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            
            V_fft_gated = gates * V_fft
            
            # Optional low-rank update
            if self.low_rank is not None:
                V_fft_gated = self._apply_toeplitz(V_fft_gated)
            
            # Apply activation
            V_fft_gated = self.modrelu(V_fft_gated)
            
            # Inverse FFT
            V_mixed = torch.fft.irfft(V_fft_gated, n=seq_len, dim=2, norm='ortho')
        
        # Optional wavelet refinement
        if self.use_wavelet:
            # Handle mean_q shape for incremental vs batch processing
            if incremental and mean_q.dim() == 2:
                # Incremental: mean_q is [heads, d_head], add batch dim
                mean_q_batch = mean_q.unsqueeze(0)  # [1, heads, d_head]
            else:
                mean_q_batch = mean_q  # Already [batch, heads, d_head]
            
            # Compute skip probability
            skip_logits = self.skip_controller(mean_q_batch.mean(dim=-2)) + self.skip_init  # [batch, 1]
            skip_prob = torch.sigmoid(skip_logits)
            
            # Apply WRM stochastically during training, deterministically during inference
            if self.training:
                # Sample whether to apply WRM for each batch element
                apply_wrm_mask = torch.bernoulli(1 - skip_prob).squeeze(-1) > 0  # [batch]
            else:
                # Apply deterministically based on threshold
                apply_wrm_mask = skip_prob.squeeze(-1) < 0.5  # [batch]
            
            # Apply WRM only if any batch element needs it
            if apply_wrm_mask.any():
                V_refined = self.wrm(V_mixed, mean_q_batch)
                # Apply only to selected batch elements
                apply_wrm_mask = apply_wrm_mask.view(-1, 1, 1, 1)
                V_mixed = torch.where(apply_wrm_mask, V_mixed + V_refined, V_mixed)
        
        # Reshape and project output
        # V_mixed shape: [batch, heads, seq_len, d_head]
        seq_len_out = V_mixed.shape[2]
        V_mixed = V_mixed.transpose(1, 2).contiguous()  # [batch, seq_len, heads, d_head]
        V_mixed = V_mixed.view(batch, seq_len_out, self.d_model)  # [batch, seq_len, d_model]
        output = self.W_o(V_mixed)
        output = self.dropout(output)
        
        return output, cache
    
    def _apply_toeplitz(self, V_fft: torch.Tensor) -> torch.Tensor:
        """Apply Toeplitz convolution in frequency domain."""
        # Handle both batch and incremental cases
        if V_fft.dim() == 4:
            # Batch case: V_fft shape: [batch, heads, n_freq, d_head]
            n_freq = V_fft.shape[2]
            kernel_unsqueeze_dims = (0, -1)  # Add batch and d_head dimensions
        else:
            # Incremental case: V_fft shape: [heads, n_freq, d_head]
            n_freq = V_fft.shape[1]
            kernel_unsqueeze_dims = (-1,)  # Only add d_head dimension
        
        # Pad kernel to match frequency dimension
        kernel_size = self.toeplitz_kernel.shape[-1]
        if kernel_size < n_freq:
            kernel_padded = F.pad(
                self.toeplitz_kernel, 
                (0, n_freq - kernel_size)
            )
        else:
            kernel_padded = self.toeplitz_kernel[:, :n_freq]
        
        # Apply depth-wise convolution in frequency domain
        # kernel_padded shape: [n_heads, n_freq]
        # Add dimensions for broadcasting
        for dim in kernel_unsqueeze_dims:
            kernel_padded = kernel_padded.unsqueeze(dim)
        
        return V_fft + V_fft * kernel_padded


class WaveletRefinementModule(nn.Module):
    """
    Optional module to capture local details using wavelets.
    Uses Haar wavelets for simplicity and efficiency.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # MLP to generate wavelet gates
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d_head, self.d_head * 2),
            nn.GELU(),
            nn.Linear(self.d_head * 2, 1)  # Per-channel gates
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(self.d_head)
    
    def forward(self, x: torch.Tensor, mean_q: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet refinement.
        
        Args:
            x: Input tensor [batch, heads, seq_len, d_head]
            mean_q: Mean query [batch, heads, d_head]
            
        Returns:
            Refined tensor
        """
        batch, heads, seq_len, d_head = x.shape
        
        # Skip wavelet transform for single token (incremental decoding)
        if seq_len == 1:
            return torch.zeros_like(x)
        
        # Apply Haar DWT (simple averaging and differencing)
        x_dwt = self._haar_dwt_1d(x)
        
        # Generate channel-wise gates from mean query
        gates = self.gate_mlp(self.norm(mean_q))  # [batch, heads, 1]
        gates = torch.sigmoid(gates).unsqueeze(-1)  # [batch, heads, 1, 1]
        
        # Apply gates
        x_dwt_gated = x_dwt * gates
        
        # Inverse DWT
        x_refined = self._haar_idwt_1d(x_dwt_gated)
        
        return x_refined[:, :, :seq_len]  # Ensure output matches input length
    
    def _haar_dwt_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Simple Haar wavelet transform."""
        # Ensure even length
        if x.shape[2] % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        
        # Averaging and differencing
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        
        low = (x_even + x_odd) / math.sqrt(2)
        high = (x_even - x_odd) / math.sqrt(2)
        
        return torch.cat([low, high], dim=2)
    
    def _haar_idwt_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse Haar wavelet transform."""
        seq_len = x.shape[2]
        mid = seq_len // 2
        
        low = x[:, :, :mid]
        high = x[:, :, mid:]
        
        # Reconstruct
        x_even = (low + high) / math.sqrt(2)
        x_odd = (low - high) / math.sqrt(2)
        
        # Interleave
        output = torch.zeros_like(x)
        output[:, :, 0::2] = x_even
        output[:, :, 1::2] = x_odd
        
        return output


class SPECTREBlock(nn.Module):
    """
    Complete SPECTRE transformer block with layer norm, residual connections, and FFN.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 8192,
        low_rank: Optional[int] = None,
        use_wavelet: bool = False,
        share_gates: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # SPECTRE layer
        self.spectre = SPECTRELayer(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            low_rank=low_rank,
            use_wavelet=use_wavelet,
            share_gates=share_gates,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[PrefixFFTCache] = None,
        incremental: bool = False,
        position: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[PrefixFFTCache]]:
        """Forward pass with residual connections."""
        # SPECTRE with residual
        normed = self.norm1(x)
        spectre_out, cache = self.spectre(normed, cache, incremental, position)
        x = x + spectre_out
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x, cache


class PersistentMemory(nn.Module):
    """
    Optional persistent memory bank that is prepended to every sequence.
    Never evicted from cache, provides long-term context.
    """
    def __init__(self, n_mem_tokens: int, d_model: int):
        super().__init__()
        self.n_mem_tokens = n_mem_tokens
        self.d_model = d_model
        
        # Learnable memory tokens
        self.memory = nn.Parameter(torch.randn(n_mem_tokens, d_model) * 0.02)
        
        # Pre-computed FFT of memory (updated in forward)
        self.register_buffer('memory_fft', torch.zeros(
            n_mem_tokens // 2 + 1, d_model, dtype=torch.complex64
        ))
    
    def forward(self) -> torch.Tensor:
        """Return memory tokens."""
        return self.memory
    
    def update_fft(self):
        """Update pre-computed FFT of memory."""
        with torch.no_grad():
            self.memory_fft = torch.fft.rfft(self.memory, dim=0, norm='ortho')


# Helper functions for model initialization
def init_spectre_caches(
    model: nn.Module,
    max_seq_len: int,
    device: torch.device = torch.device('cuda')
) -> Dict[str, PrefixFFTCache]:
    """
    Initialize Prefix-FFT caches for all SPECTRE layers in a model.
    
    Args:
        model: Model containing SPECTRE layers
        max_seq_len: Maximum sequence length
        device: Device to place caches on
        
    Returns:
        Dictionary mapping layer names to their caches
    """
    caches = {}
    
    for name, module in model.named_modules():
        if isinstance(module, SPECTRELayer):
            cache = PrefixFFTCache(
                max_seq_len=max_seq_len,
                d_model=module.d_model,
                n_heads=module.n_heads,
                device=device
            )
            caches[name] = cache
    
    return caches


# Example usage functions
def create_spectre_vit(
    img_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    n_heads: int = 12,
    mlp_ratio: float = 4.0,
    num_classes: int = 1000,
    use_wavelet: bool = False
) -> nn.Module:
    """
    Create a Vision Transformer using SPECTRE layers.
    
    This is a simplified example showing how SPECTRE can replace
    self-attention in vision transformers.
    """
    class SpectreViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
            self.n_patches = (img_size // patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim) * 0.02)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            
            self.blocks = nn.ModuleList([
                SPECTREBlock(
                    d_model=embed_dim,
                    n_heads=n_heads,
                    d_ff=int(embed_dim * mlp_ratio),
                    max_seq_len=self.n_patches + 1,
                    use_wavelet=use_wavelet
                )
                for _ in range(depth)
            ])
            
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Patch embedding
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            
            # Add cls token
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Add position embedding
            x = x + self.pos_embed
            
            # SPECTRE blocks
            for block in self.blocks:
                x, _ = block(x)
            
            # Classification head
            x = self.norm(x)
            cls_token_final = x[:, 0]
            return self.head(cls_token_final)
    
    return SpectreViT()


def create_spectre_lm(
    vocab_size: int,
    max_seq_len: int = 8192,
    d_model: int = 768,
    n_layers: int = 12,
    n_heads: int = 12,
    d_ff: int = 3072,
    use_wavelet: bool = False,
    n_mem_tokens: int = 0
) -> nn.Module:
    """
    Create a language model using SPECTRE layers.
    
    This demonstrates how SPECTRE can be used for autoregressive
    language modeling with efficient caching.
    """
    class SpectreLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.max_seq_len = max_seq_len
            
            self.token_embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
            
            # Optional persistent memory
            self.persistent_memory = None
            if n_mem_tokens > 0:
                self.persistent_memory = PersistentMemory(n_mem_tokens, d_model)
            
            self.blocks = nn.ModuleList([
                SPECTREBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len + n_mem_tokens,
                    use_wavelet=use_wavelet
                )
                for _ in range(n_layers)
            ])
            
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            
            # Tie embeddings
            self.lm_head.weight = self.token_embed.weight
        
        def forward(
            self,
            input_ids: torch.Tensor,
            caches: Optional[Dict[str, PrefixFFTCache]] = None,
            incremental: bool = False
        ) -> Tuple[torch.Tensor, Optional[Dict[str, PrefixFFTCache]]]:
            # Token embeddings
            x = self.token_embed(input_ids)
            
            # Position embeddings
            if incremental and caches:
                # Get position from first cache
                pos = list(caches.values())[0].position
                pos_ids = torch.tensor([pos], device=x.device)
            else:
                pos = None
                pos_ids = torch.arange(x.shape[1], device=x.device)
            x = x + self.pos_embed(pos_ids).unsqueeze(0)
            
            # Prepend persistent memory if available
            if self.persistent_memory is not None:
                mem = self.persistent_memory().unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([mem, x], dim=1)
            
            # Apply SPECTRE blocks
            updated_caches = {} if caches is not None else None
            position = pos if incremental and caches else None
            for i, block in enumerate(self.blocks):
                block_cache = caches.get(f'blocks.{i}.spectre', None) if caches else None
                x, new_cache = block(x, block_cache, incremental, position)
                if updated_caches is not None and new_cache is not None:
                    updated_caches[f'blocks.{i}.spectre'] = new_cache
            
            # Final norm and output projection
            x = self.norm(x)
            
            # Remove memory tokens from output
            if self.persistent_memory is not None:
                x = x[:, self.persistent_memory.n_mem_tokens:]
            
            logits = self.lm_head(x)
            
            return logits, updated_caches
    
    return SpectreLM()


if __name__ == "__main__":
    # Test basic functionality
    print("Testing SPECTRE implementation...")
    
    # Test parameters
    batch_size = 2
    seq_len = 1024
    d_model = 512
    n_heads = 8
    
    # Create a SPECTRE layer
    layer = SPECTRELayer(
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=2048,
        low_rank=16,
        use_wavelet=True
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output, _ = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test incremental decoding
    cache = PrefixFFTCache(
        max_seq_len=2048,
        d_model=d_model,
        n_heads=n_heads,
        device=x.device
    )
    
    # Pre-fill with initial sequence
    layer(x[:1], cache=cache, incremental=False)
    
    # Incremental step
    new_token = torch.randn(1, 1, d_model)
    inc_output, _ = layer(new_token, cache=cache, incremental=True, position=seq_len)
    print(f"Incremental output shape: {inc_output.shape}")
    
    print("\nSPECTRE implementation test completed successfully!")
