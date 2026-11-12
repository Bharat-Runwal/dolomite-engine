"""Avg11 (paper App. F.2): acc_norm on arc_c, arc_e, hellaswag, openbookqa, piqa, sciq; acc on boolq, copa,
winogrande, race, lambada_openai; plus wikitext word perplexity. Usage: avg11.py <harness_results*.json | dir> ..."""
import json,sys,glob,os
NORM=['arc_challenge','arc_easy','hellaswag','openbookqa','piqa','sciq']; ACC=['boolq','copa','winogrande','race','lambada_openai']
def score(p):
    if os.path.isdir(p): p=sorted(glob.glob(p+'/**/harness_results_*.json',recursive=True))[-1]
    r=json.load(open(p))['results']
    v=[r[t]['acc_norm,none'] for t in NORM]+[r[t]['acc,none'] for t in ACC]
    return 100*sum(v)/11, r['wikitext']['word_perplexity,none']
if __name__=='__main__':
    for p in sys.argv[1:]:
        try: a,w=score(p); print(f'{a:6.2f}  {w:6.2f}  {p}')
        except Exception as e: print('  --     --   ',p,type(e).__name__)
