from __future__ import annotations
import contextlib, warnings as _w
import torch, torch.nn as nn, torch.nn.functional as F
from torch import amp

# silence cruft
for msg in (r"ComplexHalf", r"Torchinductor", r"custom_fwd", r"custom_bwd"):
    _w.filterwarnings("ignore", message=msg)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
with contextlib.suppress(AttributeError):
    torch.set_float32_matmul_precision("medium")

_AMP = torch.float16

# fp16‑FFT probe
if torch.cuda.is_available():
    try:
        torch.fft.rfft(torch.empty(8, device="cuda", dtype=torch.float16)); _FFT16 = True
    except Exception:
        _FFT16 = False
else:
    _FFT16 = False

_INV_SQRT2 = 0.7071067811865476
_pow2 = lambda n: (n & (n - 1) == 0)

def _dwt(x):
    e, o = x[..., ::2, :], x[..., 1::2, :]
    return torch.cat((e + o, e - o), -2).mul_(_INV_SQRT2)

def _idwt(y):
    h = y.size(-2) // 2
    l, hi = y[..., :h, :], y[..., h:, :]
    e, o = l + hi, l - hi
    out = torch.empty_like(y); out[..., ::2, :], out[..., 1::2, :] = e, o
    return out.mul_(_INV_SQRT2)

def _mix(v: torch.Tensor, g: torch.Tensor, L: int):
    use16 = v.dtype == torch.float16 and _FFT16 and _pow2(L)
    x = v if use16 else v.float()
    s = torch.fft.rfft(x, dim=1, norm="ortho"); s.mul_(g.unsqueeze(-1))
    y = torch.fft.irfft(s, n=L, dim=1, norm="ortho")
    return y if use16 else y.to(v.dtype)

class Spectre(nn.Module):
    __constants__ = ("d_model","n_heads","d_head","seq_len","chunk","wavelet")
    def __init__(self, d_model, n_heads, seq_len, *, chunk_size=None, gate_hidden=None, wavelet=False, fused_qkv=False):
        super().__init__()
        if d_model % n_heads: raise ValueError
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head, self.seq_len = d_model // n_heads, seq_len
        self.chunk = chunk_size or seq_len; self.wavelet = wavelet; self.fused = fused_qkv
        self.pad = (self.chunk - seq_len % self.chunk) % self.chunk
        self.n_chunks = (seq_len + self.pad) // self.chunk
        self.fF, self.fC = seq_len // 2 + 1, self.chunk // 2 + 1
        # proj
        if fused_qkv:
            self.w_qv = nn.Linear(d_model, 2*d_model, False)
        else:
            self.w_q = nn.Linear(d_model, d_model, False)
            self.w_v = nn.Linear(d_model, d_model, False)
        self.w_o = nn.Linear(d_model, d_model, False)
        gh = gate_hidden or 4*self.d_head
        self.ln_q = nn.LayerNorm(self.d_head, eps=1e-6)
        self.g_up = nn.Linear(self.d_head, gh, False)
        self.g_dnF = nn.Linear(gh, 2*self.fF, False)
        self.g_dnC = nn.Linear(gh, 2*self.fC, False)
        if wavelet:
            self.wu = nn.Linear(self.d_head, gh, False)
            self.wd = nn.Linear(gh, seq_len, False)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, 0.8)

    def _proj(self, x):
        with amp.autocast(device_type="cuda", dtype=_AMP):
            if self.fused:
                qv = self.w_qv(x); return qv.chunk(2, -1)
            return self.w_q(x), self.w_v(x)

    def forward(self, x):
        B,N,_ = x.shape
        if N != self.seq_len: raise RuntimeError
        q,v = self._proj(x)
        q = q.view(B,N,self.n_heads,self.d_head).transpose(1,2)
        v = v.view(B,N,self.n_heads,self.d_head).transpose(1,2)
        if self.chunk >= N:
            with amp.autocast(device_type="cuda", dtype=_AMP):
                bar = self.ln_q(q.mean(2)); g = self.g_dnF(F.silu(self.g_up(bar)))
            g = g.float().view(-1,self.fF,2); gc = torch.complex(g[...,0],g[...,1])
            vt = _mix(v.reshape(-1,N,self.d_head), gc, N).view(B,self.n_heads,N,self.d_head)
            if self.wavelet:
                w = _dwt(vt)
                with amp.autocast(device_type="cuda", dtype=_AMP):
                    s = self.wd(F.silu(self.wu(bar))).view(-1,N,1).to(vt.dtype)
                vt = vt + _idwt(w.mul_(s))
        else:
            if self.pad: v = F.pad(v,(0,0,0,self.pad)); q = F.pad(q,(0,0,0,self.pad))
            Nc = self.chunk
            v_ = v.view(B,self.n_heads,self.n_chunks,Nc,self.d_head)
            with amp.autocast(device_type="cuda", dtype=_AMP):
                bar = self.ln_q(q.view(B,self.n_heads,self.n_chunks,Nc,self.d_head).mean(3))
                g = self.g_dnC(F.silu(self.g_up(bar)))
            g = g.float().view(-1,self.fC,2); gc = torch.complex(g[...,0],g[...,1])
            vt = _mix(v_.reshape(-1,Nc,self.d_head), gc, Nc).view(B,self.n_heads,self.n_chunks,Nc,self.d_head).reshape(B,self.n_heads,N+self.pad,self.d_head)[...,:N,:]
        with amp.autocast(device_type="cuda", dtype=_AMP):
            y = vt.transpose(1,2).reshape(B,N,self.d_model)
            return self.w_o(y)
