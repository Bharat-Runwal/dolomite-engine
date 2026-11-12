import re,glob,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
L='/proj/dmfexp/energy-gpt/logs/rebuttal/'
runs=[('12S (0 energy blocks)',glob.glob(L+'reb_12S_isoact_d768-*.err')[0],'#2a78d6','o'),
      ('8S4E (4)',glob.glob(L+'reb_8S4E_isoact_d768-*.err')[0],'#eb6834','s'),
      ('6S6E (6)','/u/bsaha3/bsub_logs/fineweb_G6_6S6E_isoact_d768_s7_1921758.stderr','#1baf7a','^'),
      ('12E (12)',glob.glob(L+'reb_12E_isoact_d768-*.err')[0],'#eda100','D')]
TPS=524288
fig,ax=plt.subplots(figsize=(6.4,4.0),dpi=200); ends=[]
for name,f,c,m in runs:
    d={}
    for s,v in re.findall(r'step = (\d+), train-loss = [0-9.]+, train-lm_loss = ([0-9.]+)',open(f).read()): d[int(s)]=float(v)
    st=np.array(sorted(d)); y=np.array([d[k] for k in st]); k=25; ys=np.convolve(y,np.ones(k)/k,mode='valid'); xs=st[k-1:]*TPS/1e9
    ax.plot(xs,ys,color=c,lw=2,label=name); ax.plot(xs[-1],ys[-1],marker=m,ms=8,color=c,mec='white',mew=1.5)
    ends.append((ys[-1],xs[-1]))
ax.set_xscale('log'); ax.set_ylim(3.1,4.6); ax.set_xlim(0.2,6.0)
ys_=sorted(ends); lab=[]
for y,x in ys_:
    yl=max(y,lab[-1]+0.045) if lab else y; lab.append(yl); ax.annotate(f'{y:.3f}',(x,y),xytext=(x*1.07,yl),va='center',fontsize=8,color='#52514e')
ax.set_xlabel('training tokens (B)',color='#52514e'); ax.set_ylabel('LM loss (nats, smoothed)',color='#52514e')
ax.set_title('Depth-matched ladder at d=768: 12 layers, 137M active, 4.0B tokens',fontsize=10,loc='left')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['left','bottom']: ax.spines[sp].set_color('#c3c2b7')
from matplotlib.ticker import FixedLocator,NullLocator,FixedFormatter
ax.xaxis.set_major_locator(FixedLocator([0.25,0.5,1,2,4])); ax.xaxis.set_major_formatter(FixedFormatter(['0.25','0.5','1','2','4'])); ax.xaxis.set_minor_locator(NullLocator())
ax.tick_params(colors='#52514e',labelsize=8); ax.grid(True,which='major',color='#e6e5e0',lw=0.6)
ax.legend(title='model (energy blocks)',fontsize=8,title_fontsize=8,frameon=False,loc='upper right')
fig.tight_layout(); fig.savefig('/proj/dmfexp/energy-gpt/checkpoints-bsaha/rebuttal/plots/ladder_loss.png'); fig.savefig('/proj/dmfexp/energy-gpt/checkpoints-bsaha/rebuttal/plots/ladder_loss.pdf')
print('ok')
