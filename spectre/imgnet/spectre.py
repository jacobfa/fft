from __future__ import annotations
import math, warnings as _w, contextlib
import torch, torch.nn as nn, torch.nn.functional as F
from torch import amp

# ─── silence noisy warnings ───────────────────────────────────────────────────
for msg in (
    r"ComplexHalf support is experimental",
    r"Torchinductor does not support code generation",
    r"`torch\.cuda\.amp\.custom_fwd",
    r"`torch\.cuda\.amp\.custom_bwd",
):
    _w.filterwarnings("ignore", message=msg)

# ─── global perf knobs ────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
with contextlib.suppress(AttributeError):
    torch.set_float32_matmul_precision("medium")    # TF32 without fp32 slowdown

_AMP_DTYPE = torch.float16

# ─── float‑16 FFT support probe ───────────────────────────────────────────────

def _detect_fft_supported() -> set[torch.dtype]:
    ok = {torch.float32}
    if torch.cuda.is_available():
        try:
            torch.fft.rfft(torch.empty(8, device="cuda", dtype=torch.float16))
            ok.add(torch.float16)
        except Exception:
            pass
    return ok

_FFT_OK = _detect_fft_supported()

# ─── helpers ──────────────────────────────────────────────────────────────────

def _is_power_of_two(n: int) -> bool:
    return n and (n & (n - 1) == 0)

def _next_pow_two(n: int) -> int:
    return 1 << (n - 1).bit_length()

# ─── fast DropPath (no mask allocation in eval) ───────────────────────────────
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = float(p)

    def forward(self, x: torch.Tensor):
        if self.p == 0.0 or not self.training:
            return x
        return F.dropout(x, self.p, self.training, inplace=True)

# ─── Haar helpers ─────────────────────────────────────────────────────────────
_INV_SQRT2 = 0.7071067811865476

def _dwt_haar(x: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., ::2, :], x[..., 1::2, :]
    return torch.cat((even + odd, even - odd), -2).mul_(_INV_SQRT2)

def _idwt_haar(y: torch.Tensor) -> torch.Tensor:
    half = y.size(-2) // 2
    low, high = y[..., :half, :], y[..., half:, :]
    even, odd = low + high, low - high
    out = torch.empty_like(y)
    out[..., ::2, :], out[..., 1::2, :] = even, odd
    return out.mul_(_INV_SQRT2)

# ─── FFT mixer kernel (pure eager) ────────────────────────────────────────────

def _mix_chunk(v: torch.Tensor, gate_c: torch.Tensor, L: int):
    """FFT‑based mixing that keeps fp16 kernels whenever possible.

    * If fp16 FFT is supported **and** *L* is a power‑of‑two → run natively.
    * Else pad/trim in the frequency domain so shapes always match.
    * Falls back to fp32 only as a last resort.
    """
    dt = v.dtype

    # ── Case 1: native fp16 (or fp32) with power‑of‑two length ───────────────
    if _is_power_of_two(L) and dt in _FFT_OK:
        spec = torch.fft.rfft(v, dim=1, norm="ortho")
        spec *= gate_c.unsqueeze(-1)
        return torch.fft.irfft(spec, n=L, dim=1, norm="ortho")

    # ── Case 2: stay in fp16 by zero‑padding to next pow‑two ─────────────────
    if dt is torch.float16 and torch.float16 in _FFT_OK:
        P = _next_pow_two(L)
        pad_len = P - L
        vin = F.pad(v, (0, 0, 0, pad_len))  # pad length‑dim (dim=1) right side

        freq_target = P // 2 + 1
        diff = freq_target - gate_c.size(1)
        if diff > 0:
            gate_pad = F.pad(gate_c, (0, diff))
        elif diff < 0:
            gate_pad = gate_c[..., :freq_target]
        else:
            gate_pad = gate_c

        spec = torch.fft.rfft(vin, dim=1, norm="ortho")
        spec *= gate_pad.unsqueeze(-1)
        out = torch.fft.irfft(spec, n=P, dim=1, norm="ortho")
        return out[:, :L, :]

    # ── Case 3: fp32 fallback ────────────────────────────────────────────────
    vin = v.float()
    spec = torch.fft.rfft(vin, dim=1, norm="ortho")
    spec *= gate_c.unsqueeze(-1)
    out = torch.fft.irfft(spec, n=L, dim=1, norm="ortho")
    return out.to(dt)

# ─── Spectre mixer (compile‑free) ─────────────────────────────────────────────
class Spectre(nn.Module):
    __constants__ = ("d_model", "n_heads", "d_head", "seq_len", "chunk", "wavelet")

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        *,
        chunk_size: int | None = None,
        gate_hidden: int | None = None,
        wavelet: bool = False,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head, self.seq_len  = d_model // n_heads, seq_len
        self.chunk = chunk_size or seq_len
        if self.chunk < 8:
            raise ValueError("chunk_size must be at least 8")
        self.wavelet = wavelet

        self.pad      = (self.chunk - seq_len % self.chunk) % self.chunk
        self.n_chunks = (seq_len + self.pad) // self.chunk
        self.freq_f, self.freq_c = seq_len // 2 + 1, self.chunk // 2 + 1

        # projections
        self.w_q, self.w_v, self.w_o = (
            nn.Linear(d_model, d_model, False) for _ in range(3)
        )

        gh = gate_hidden or 4 * self.d_head
        self.ln_q      = nn.LayerNorm(self.d_head, 1e-6)
        self.g_up      = nn.Linear(self.d_head, gh, False)
        self.g_dn_full = nn.Linear(gh, 2 * self.freq_f,  False)
        self.g_dn_chunk= nn.Linear(gh, 2 * self.freq_c,  False)

        if wavelet:
            self.wrm_up   = nn.Linear(self.d_head, gh, False)
            self.wrm_down = nn.Linear(gh, seq_len, False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)

    # --------------------------------------------------------------------- fwd
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        if N != self.seq_len:
            raise RuntimeError(f"Spectre expected seq_len={self.seq_len}, got {N}")

        # == projections (AMP) ==================================================
        with amp.autocast(device_type="cuda", dtype=_AMP_DTYPE):
            q, v = self.w_q(x), self.w_v(x)

        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # full‑sequence path ----------------------------------------------------
        if self.chunk >= N:
            bar     = self.ln_q(q.mean(2))                       # (B,H,Dh)
            gate_rt = self.g_dn_full(F.silu(self.g_up(bar))).float()
            gate_rt = gate_rt.view(-1, self.freq_f, 2)
            gate_c  = torch.complex(gate_rt[..., 0], gate_rt[..., 1])

            out = _mix_chunk(v.reshape(-1, N, self.d_head), gate_c, N)
            v_t = out.view(B, self.n_heads, N, self.d_head)

            if self.wavelet:
                w  = _dwt_haar(v_t)
                s  = self.wrm_down(F.silu(self.wrm_up(bar))).view(-1, N, 1).to(v_t.dtype)
                v_t = v_t + _idwt_haar(w.mul_(s))

        # chunked path ----------------------------------------------------------
        else:
            if self.pad:
                v = F.pad(v, (0, 0, 0, self.pad))
                q = F.pad(q, (0, 0, 0, self.pad))

            Nc  = self.chunk
            v   = v.view(B, self.n_heads, self.n_chunks, Nc, self.d_head)
            bar = self.ln_q(q.view(B, self.n_heads, self.n_chunks, Nc, self.d_head).mean(3))

            gate_rt = self.g_dn_chunk(F.silu(self.g_up(bar))).float()
            gate_rt = gate_rt.view(-1, self.freq_c, 2)
            gate_c  = torch.complex(gate_rt[..., 0], gate_rt[..., 1])

            out = _mix_chunk(v.reshape(-1, Nc, self.d_head), gate_c, Nc)
            v_t = (
                out.view(B, self.n_heads, self.n_chunks, Nc, self.d_head)
                .reshape(B, self.n_heads, N + self.pad, self.d_head)[..., :N, :]
            )

        # output proj (AMP) -----------------------------------------------------
        with amp.autocast(device_type="cuda", dtype=_AMP_DTYPE):
            y = v_t.transpose(1, 2).reshape(B, N, self.d_model)
            return self.w_o(y)
