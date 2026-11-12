import yaml,copy,sys,os
sys.path.insert(0,'/proj/dmfexp/bishwajit/Code/de-reb')
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import audit_config
R='/proj/dmfexp/bishwajit/Code/de-reb/configs/iclr_26/'
base=yaml.safe_load(open(R+'fwe_sweep/fineweb_G6_6S6E_isoact_d768.yml'))
pc=base['model_args']['pretrained_config']
SA,EA=pc['sequence_mixer_blocks'][0],pc['sequence_mixer_blocks'][-1]
SM,EM=pc['mlp_blocks'][0],pc['mlp_blocks'][-1]
def write(name,pattern,ws,we,path):
    c=copy.deepcopy(base); p=c['model_args']['pretrained_config']
    sm=dict(SM,intermediate_size=ws); em=dict(EM,intermediate_size=we*EM['n_experts'])
    p['sequence_mixer_blocks']=[dict(SA if t=='S' else EA) for t in pattern]
    p['mlp_blocks']=[dict(sm if t=='S' else em) for t in pattern]
    c['datasets'][0]['class_args']['data_cache_path']='/proj/dmfexp/energy-gpt/.cache/megatron_fwe_nemotron_bsaha'
    c['save_args']['save_path']=f'/proj/dmfexp/energy-gpt/checkpoints-bsaha/rebuttal/{name}'
    c['experiments_tracker_args'] if 'experiments_tracker_args' in c else None
    c['logging_args']['wandb_args']['name']=name
    c['training_parameters']['gradient_accumulation_steps']=8
    yaml.safe_dump(c,open(path,'w'),sort_keys=False,width=200)
    return audit_config(path)
g6=audit_config(R+'fwe_sweep/fineweb_G6_6S6E_isoact_d768.yml'); print('G6 total %.1fM active %.1fM'%(g6.total/1e6,g6.active/1e6))
e3=audit_config(R+'fwe_sweep/fineweb_E3_switch_K8.yml'); print('E3 total %.1fM active %.1fM'%(e3.total/1e6,e3.active/1e6))
tgt=g6.active
for name,pat in [('reb_12S_isoact_d768','S'*12),('reb_12E_isoact_d768','E'*12),('reb_8S4E_isoact_d768','S'*8+'E'*4)]:
    path=R+'bsaha/rebuttal/'+name+'.yml'
    # same expert-width RATIO as G6 (608 swiglu : 912 w1w2), scale both by f to hit G6 active
    lo,hi=0.3,3.0
    for _ in range(40):
        f=(lo+hi)/2; a=write(name,pat,round(608*f),round(912*f),path)
        if a.active<tgt: lo=f
        else: hi=f
    a=write(name,pat,round(608*f),round(912*f),path)
    print(f'{name}: S width {round(608*f)} E width/expert {round(912*f)}  total {a.total/1e6:.1f}M active {a.active/1e6:.1f}M (target {tgt/1e6:.1f}M)')
