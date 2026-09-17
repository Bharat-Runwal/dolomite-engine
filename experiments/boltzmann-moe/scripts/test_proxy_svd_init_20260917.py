"""Does an SVD warm start beat the DISTILLED proxy at ranking experts?

The proxy computes  E_hat_k = mean(gelu(B_k V_k^T x)^2)*s_k + b_k,  which is the true energy
form  A_k = mean(gelu(W_k x)^2)  with W_k (I_e,d) replaced by the factored pair B_k V_k^T
(B_k: (m,r), V_k: (d,r)).  Today V and B are torch.randn and are shaped ONLY by distillation.
So the obvious warm start is the actual rank-r factorisation of W_k:
    W_k = U S V^T   ->   V_k := V[:, :r]      (d,r)
                         B_k := (U S)[rows, :r]  subsampled to m rows   (m,r)
Rows are chosen by largest norm (the output dims that carry the most energy), and proxy_scale
absorbs the mean-over-m vs mean-over-I_e mismatch.

CAVEAT this measures: SVD is optimal for ||W_k x|| in Frobenius norm, NOT for RANKING
mean(gelu(.)^2) across experts. So it is a warm start, not a replacement for distillation.
Reported: top-k set agreement against the EXACT energies, for (a) the trained proxy,
(b) SVD init, (c) random re-init as a floor.
"""
import sys, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_engine.hf_models import register_model_classes
register_model_classes()

ck = sys.argv[1]; tokp = "/proj/datasets/tokenizers/granite-4.0-tiktoken"
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.float32, trust_remote_code=True).cuda().eval()
tok = AutoTokenizer.from_pretrained(tokp, trust_remote_code=True)
ids = tok((open("README.md").read()+open("lm_engine/pretrain.py").read())*2, return_tensors="pt").input_ids[:, :4096].cuda()

moes = [m for m in model.modules() if hasattr(m, "proxy_V") and getattr(m, "proxy_V", None) is not None]
print(f"{len(moes)} MoE blocks with a proxy; K={moes[0].n_experts} r={moes[0].proxy_V.shape[-1]} "
      f"m={moes[0].proxy_B.shape[-2]} I_e={moes[0]._expert_I}")

def agreement(mod, x):
    """|top-k(proxy) ∩ top-k(exact)| / k, per token, averaged."""
    with torch.no_grad():
        E_hat = mod._proxy_energies(x).reshape(-1, mod.n_experts)
        W = mod._fused_W().view(mod.n_experts, mod._expert_I, mod.hidden_size).float()
        xf = x.reshape(-1, mod.hidden_size).float()
        gz = F.gelu(torch.einsum("td,kid->tki", xf, W))
        E_ex = (gz*gz).mean(-1)                                    # (T,K) exact
        k = int(mod.top_k)
        a = E_hat.topk(k, -1).indices; b = E_ex.topk(k, -1).indices
        return (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1).mean().item()/k

def svd_init(mod):
    with torch.no_grad():
        W = mod._fused_W().view(mod.n_experts, mod._expert_I, mod.hidden_size).float()
        r = mod.proxy_V.shape[-1]; m = mod.proxy_B.shape[-2]
        for k in range(mod.n_experts):
            U, S, Vh = torch.linalg.svd(W[k], full_matrices=False)
            mod.proxy_V.data[k] = Vh[:r].T.to(mod.proxy_V.dtype)          # (d,r)
            US = U[:, :r] * S[:r]                                          # (I_e,r)
            rows = US.norm(dim=-1).topk(min(m, US.shape[0])).indices        # most energetic dims
            mod.proxy_B.data[k].zero_()
            mod.proxy_B.data[k][:rows.numel()] = US[rows].to(mod.proxy_B.dtype)
            mod.proxy_scale.data[k] = 1.0; mod.proxy_bias.data[k] = 0.0

# capture each block's input
xs = {}
hooks = [m.register_forward_pre_hook(lambda mod, inp, _i=i: xs.__setitem__(_i, inp[0].detach()))
         for i, m in enumerate(moes)]
with torch.no_grad(): model(input_ids=ids)
for h in hooks: h.remove()

saved = [(m.proxy_V.data.clone(), m.proxy_B.data.clone(), m.proxy_scale.data.clone(), m.proxy_bias.data.clone()) for m in moes]
print(f"\n{'block':>6} {'trained':>9} {'SVD init':>9} {'random':>9}")
tot = [0.0, 0.0, 0.0]
for i, mod in enumerate(moes):
    x = xs[i]
    a_tr = agreement(mod, x)
    svd_init(mod);                     a_svd = agreement(mod, x)
    with torch.no_grad():
        mod.proxy_V.data.normal_(0, mod.hidden_size**-0.5)
        mod.proxy_B.data.normal_(0, mod.proxy_B.shape[-1]**-0.5)
        mod.proxy_scale.data.fill_(1.0); mod.proxy_bias.data.zero_()
    a_rnd = agreement(mod, x)
    V,B,S,Bi = saved[i]; mod.proxy_V.data.copy_(V); mod.proxy_B.data.copy_(B)
    mod.proxy_scale.data.copy_(S); mod.proxy_bias.data.copy_(Bi)
    print(f"{i:>6} {a_tr:>9.4f} {a_svd:>9.4f} {a_rnd:>9.4f}")
    tot = [tot[0]+a_tr, tot[1]+a_svd, tot[2]+a_rnd]
n = len(moes)
print(f"{'MEAN':>6} {tot[0]/n:>9.4f} {tot[1]/n:>9.4f} {tot[2]/n:>9.4f}")
print(f"\nSVD init vs trained: {tot[1]/n - tot[0]/n:+.4f}   vs random floor: {tot[1]/n - tot[2]/n:+.4f}")
