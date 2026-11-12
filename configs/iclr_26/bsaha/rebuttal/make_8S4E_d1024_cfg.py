import yaml,copy,sys
sys.path.insert(0,'/proj/dmfexp/bishwajit/Code/de-reb')
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import audit_config
R='/proj/dmfexp/bishwajit/Code/de-reb/configs/iclr_26/'
B=R+'fwe_sweep/fineweb_G7_6S6E_isoact_d1024.yml'
base=yaml.safe_load(open(B)); pc=base['model_args']['pretrained_config']
SA,EA=pc['sequence_mixer_blocks'][0],pc['sequence_mixer_blocks'][-1]; SM,EM=pc['mlp_blocks'][0],pc['mlp_blocks'][-1]
ws0,we0=SM['intermediate_size'],EM['intermediate_size']//EM['n_experts']
g7=audit_config(B); s1=audit_config(R+'fwe_sweep/fineweb_S1_12S_isoact_d1024.yml')
print('G7 %.1f/%.1f  S1 12S %.1f/%.1f'%(g7.total/1e6,g7.active/1e6,s1.total/1e6,s1.active/1e6), 'base widths',ws0,we0)
name='reb_8S4E_isoact_d1024'; path=R+'bsaha/rebuttal/'+name+'.yml'; pat='S'*8+'E'*4
def write(f):
    c=copy.deepcopy(base); p=c['model_args']['pretrained_config']
    sm=dict(SM,intermediate_size=round(ws0*f)); em=dict(EM,intermediate_size=round(we0*f)*EM['n_experts'])
    p['sequence_mixer_blocks']=[dict(SA if t=='S' else EA) for t in pat]; p['mlp_blocks']=[dict(sm if t=='S' else em) for t in pat]
    c['datasets'][0]['class_args']['data_cache_path']='/proj/dmfexp/energy-gpt/.cache/megatron_fwe_nemotron_bsaha'
    c['save_args']['save_path']='/proj/dmfexp/energy-gpt/checkpoints-bsaha/rebuttal/'+name
    c['logging_args']['wandb_args']['name']=name
    yaml.safe_dump(c,open(path,'w'),sort_keys=False,width=200); return audit_config(path)
tgt=s1.active; lo,hi=0.3,3
for _ in range(40):
    f=(lo+hi)/2; a=write(f)
    if a.active<tgt: lo=f
    else: hi=f
a=write(f); print(name,'S',round(ws0*f),'E/expert',round(we0*f),'total %.1f active %.1f'%(a.total/1e6,a.active/1e6))
