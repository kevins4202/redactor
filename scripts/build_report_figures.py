"""Report charts drawn directly from the saved experiment JSON."""
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/latex/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
colors = ['#7a8590', '#d18a39', '#267563']
methods = ['lasso', 'classical', 'sam2']
labels = ['Original', 'Our classical\ntracker', 'SAM 2']
fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.5), layout='constrained')
for col, (dataset,title) in enumerate([('synthetic','Test scenes'),('davis','DAVIS videos'),('crosswalk','Crosswalk')]):
    report = json.loads((ROOT/f'results/final-{dataset}.json').read_text())
    assert report['complete']
    groups = [[r for r in report['runs'] if r['method']==m] for m in methods]
    for row,key in enumerate(['mean_iou','mean_leak_rate']):
        ax=axes[row,col]
        values=[st.mean(r[key] for r in g)*(100 if row else 1) for g in groups]
        ax.bar(range(3),values,color=colors,width=.6)
        ax.set_xticks(range(3),labels,fontsize=9)
        ax.set_ylim(0,85 if row else 1)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for i,v in enumerate(values): ax.text(i,v+(2 if row else .025),f'{v:.1f}%' if row else f'{v:.3f}',ha='center',fontsize=10)
        if row==0: ax.set_title(title,fontweight='bold')
        if col==0: ax.set_ylabel('Mask overlap (IoU)\nHigher is better' if row==0 else 'Target pixels left uncovered (%)\nLower is better')
fig.savefig(OUT/'results.pdf',bbox_inches='tight');fig.savefig(OUT/'results.png',dpi=180,bbox_inches='tight');plt.close(fig)

report=json.loads((ROOT/'results/coverage-davis.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(9.8,2.7),layout='constrained')
for ax,key,title,color in zip(axes,['mean_leak_rate','mean_excess_frame_fraction'],['Target left uncovered','Extra background covered'],['#b34d43','#267563']):
    vals=[100*st.mean(r['coverage_sweep'][str(2*m+1)][key] for r in report['runs']) for m in [4,8,16,32]]
    ax.plot([4,8,16,32],vals,'o-',color=color,lw=2)
    ax.set_xticks([4,8,16,32]);ax.set_xlabel('Added margin on each side (pixels)')
    ax.set_ylabel('% of target' if key=='mean_leak_rate' else '% of whole frame')
    ax.set_title(title,fontweight='bold');ax.set_ylim(0,max(vals)*1.3);ax.grid(alpha=.15)
    for x,v in zip([4,8,16,32],vals):ax.annotate(f'{v:.2f}%',(x,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
fig.savefig(OUT/'margin.pdf',bbox_inches='tight');fig.savefig(OUT/'margin.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('Wrote result and margin charts from saved JSON.')

report=json.loads((ROOT/'results/final-synthetic.json').read_text())
names=['box','lasso','gate','affine','classical','sam2']
rows=['Box start','Original outline','Flow gate','Scale + refinement','Recovery added','SAM 2']
scenarios=['translation','scale','occlusion','cut']
matrix=np.array([[next(r['mean_iou'] for r in report['runs'] if r['method']==m and r['sequence']==s and r['slack']==5) for s in scenarios] for m in names])
fig,ax=plt.subplots(figsize=(9.5,4.1),layout='constrained')
ax.imshow(matrix,cmap='YlGn',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(4),['Move','Shrink','Hide behind a pillar','Scene cut'])
ax.set_yticks(range(6),rows);ax.tick_params(length=0)
for y in range(6):
    for x in range(4):ax.text(x,y,f'{matrix[y,x]:.3f}',ha='center',va='center',color='white' if matrix[y,x]>.65 else '#17251f',fontsize=13)
ax.set_title('Mask overlap at 5-pixel prompt slack (higher is better)',pad=14)
fig.savefig(OUT/'iterations.pdf',bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9.5,3.1),layout='constrained')
report=json.loads((ROOT/'results/recovery-davis.json').read_text())
for ax,key,title in zip(axes,['mean_iou','mean_leak_rate'],['Mask overlap (higher is better)','Uncovered target (lower is better)']):
    v=[st.mean(r[key] for r in report['runs'] if r['method']==m)*(100 if key=='mean_leak_rate' else 1) for m in ['classical','learned']]
    ax.bar([0,1],v,color=['#d18a39','#7d719c'],width=.5)
    ax.set_xticks([0,1],['Template recovery','ResNet recovery']);ax.set_title(title,fontsize=11)
    ax.set_ylim(0,.4 if key=='mean_iou' else 65)
    for i,a in enumerate(v):ax.text(i,a+(.008 if key=='mean_iou' else 1),f'{a:.3f}' if key=='mean_iou' else f'{a:.2f}%',ha='center')
fig.savefig(OUT/'recovery.pdf',bbox_inches='tight');plt.close(fig)
