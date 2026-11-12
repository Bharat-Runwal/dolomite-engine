"""Per-block energy on held-out text for the depth ladder (8S4E, 6S6E, 12E; 12S has no energy blocks).
For every energy block l (sequence_mixer_type == energy_attention) we hook its forward and record, per token,
  E_l(h_in)  and  E_l(h_out)     with  E_l = E_attn + scale_ff * E_ff  (block.energy_per_token),
plus the attention and FF parts separately. Writes a json per model.
Usage: python energy_probe.py <hf_model_dir> <out.json> [n_seq] [seq_len]"""
import json,sys,torch,os
# same model registration + loading patches the eval harness uses (lm_engine.hf_models registers model_type 'energy')
_src=open(os.path.join(os.path.dirname(__file__),'../../../eval_scripts/eval_harness.py')).read()
exec(compile(_src.split('# ── Run lm_eval CLI')[0],'eval_harness_patches','exec'))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
M,OUT=sys.argv[1],sys.argv[2]; NS=int(sys.argv[3]) if len(sys.argv)>3 else 32; L=int(sys.argv[4]) if len(sys.argv)>4 else 1024
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.float32,trust_remote_code=True).cuda().eval()
txt="\n\n".join(load_dataset("wikitext","wikitext-2-raw-v1",split="test")["text"])
ids=torch.tensor(tok(txt)["input_ids"][:NS*L]).view(NS,L)
blocks=[(i,b) for i,b in enumerate(model.transformer.h) if getattr(b,'sequence_mixer_type','')=='energy_attention']
rec={i:{'E_in':[],'E_out':[],'Ea_in':[],'Ef_in':[],'Ea_out':[],'Ef_out':[],'upd_norm':[],'h_norm':[]} for i,_ in blocks}
cap={}
def pre(i):
    def f(mod,args,kw): cap[i]=(args[0] if args else kw['hidden_states'], kw.get('rope_cos_sin', args[3] if len(args)>3 else None))
    return f
def post(i):
    def f(mod,args,kw,out):
        x,r=cap[i]; y=out[0] if isinstance(out,tuple) else out
        with torch.no_grad():
            for tag,h in (('in',x),('out',y)):
                ln=mod.ln(h); ea=mod.attn.energy_per_token(ln,rope_cos_sin=r).float(); ef=(mod.scale_ff*mod.ffwd.energy_per_token(ln)).float()
                rec[i]['Ea_'+tag].append(ea.mean().item()); rec[i]['Ef_'+tag].append(ef.mean().item()); rec[i]['E_'+tag].append((ea+ef).mean().item())
            rec[i]['upd_norm'].append(((y-x).norm(dim=-1)/x.norm(dim=-1)).mean().item()); rec[i]['h_norm'].append(x.norm(dim=-1).mean().item())
    return f
for i,b in blocks: b.register_forward_pre_hook(pre(i),with_kwargs=True); b.register_forward_hook(post(i),with_kwargs=True)
with torch.no_grad():
    for k in range(0,NS,4): model(input_ids=ids[k:k+4].cuda())
res={str(i):{k:sum(v)/len(v) for k,v in d.items()} for i,d in rec.items()}
json.dump({'model':M,'n_tokens':NS*L,'blocks':res},open(OUT,'w'),indent=1); print(json.dumps(res,indent=1))
