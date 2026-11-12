import json,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P='/proj/dmfexp/energy-gpt/checkpoints-bsaha/rebuttal/energy_probe/'
M=[('8S4E','#eb6834'),('6S6E','#1baf7a'),('12E','#eda100')]
fig,axs=plt.subplots(2,3,figsize=(10,5.2),dpi=200,gridspec_kw={'height_ratios':[3,2]},sharex=True)
for j,(m,c) in enumerate(M):
    b=json.load(open(P+m+'.json'))['blocks']; L=sorted(int(k) for k in b)
    ei=[b[str(l)]['E_in'] for l in L]; eo=[b[str(l)]['E_out'] for l in L]
    ax=axs[0,j]
    for l,a,o in zip(L,ei,eo): ax.annotate('',xy=(l,o),xytext=(l,a),arrowprops=dict(arrowstyle='-|>',color=c,lw=2,mutation_scale=10))
    ax.plot(L,ei,'o',mfc='white',mec=c,mew=2,ms=8,label='E$_l$ at block input'); ax.plot(L,eo,'o',color=c,mec='white',mew=1.5,ms=8,label='E$_l$ at block output')
    ax.set_title(f'{m}  ({len(L)} energy blocks)',fontsize=10,loc='left')
    d=[o-a for a,o in zip(ei,eo)]; ax2=axs[1,j]
    ax2.bar(L,d,width=0.6,color=c); ax2.axhline(0,color='#52514e',lw=0.8)
    for l,v in zip(L,d): ax2.annotate(f'{v:+.2f}',(l,v),xytext=(0,3 if v>=0 else -9),textcoords='offset points',ha='center',fontsize=7,color='#52514e')
    for a in (ax,ax2):
        for sp in ['top','right']: a.spines[sp].set_visible(False)
        for sp in ['left','bottom']: a.spines[sp].set_color('#c3c2b7')
        a.tick_params(colors='#52514e',labelsize=8); a.grid(True,axis='y',color='#e6e5e0',lw=0.6)
    ax2.set_xlabel('layer index',color='#52514e',fontsize=9); ax2.set_xticks(range(0,12))
axs[0,0].set_ylabel('block energy per token',color='#52514e',fontsize=9); axs[1,0].set_ylabel('$\\Delta E_l$ = out $-$ in\n(<0: descent)',color='#52514e',fontsize=9)
axs[0,0].legend(fontsize=7,frameon=False,loc='lower left')
fig.suptitle("Each energy block's own energy before and after its update (held-out WikiText, 32k tokens, deployed router). 12S has no energy blocks.",fontsize=9,x=0.01,ha='left')
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(P+'../plots/ladder_energy.png'); fig.savefig(P+'../plots/ladder_energy.pdf'); print('ok')
