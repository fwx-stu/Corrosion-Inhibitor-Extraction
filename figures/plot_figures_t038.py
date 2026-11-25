"""Draw eight data-backed manuscript figures with Matplotlib (600 dpi RGB TIFF)."""
from __future__ import annotations
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

WIDTH = 7.09  # 180 mm at final manuscript width; no tight-bbox resizing.
NAVY, TEAL, ORANGE = '#1E3A5E', '#018F87', '#ED6B16'
GRID = '#E3E6EA'
FACES = ['#BCD3DC', '#CBDFC6', '#FDE8CA']
EDGES = ['#64969E', '#768A72', '#AD906A']
STRATEGIES = ['q0_safe_full', 'q1_budgeted_rag', 'q2_table_aware_rag']
MODELS = ['qwen2_5_3b', 'qwen2_5_7b', 'qwen2_5_14b']
METRICS = ['field_micro_f1', 'record_f1', 'null_accuracy']
NAMES = {'qwen2_5_3b':'Qwen2.5-3B','qwen2_5_7b':'Qwen2.5-7B',
         'qwen2_5_14b':'Qwen2.5-14B','glm_4_plus':'GLM-4-Plus'}
FIELDS = ['inhibitor_name','substrate','corrosive_medium','medium_concentration',
          'inhibitor_concentration','temperature','immersion_time','test_method',
          'inhibition_efficiency','homo','lumo','delta_g_ads','adsorption_energy']
FIELD_LABELS = ['Inhibitor name','Substrate','Corrosive medium','Medium concentration',
                'Inhibitor concentration','Temperature','Immersion time','Test method',
                'Inhibition efficiency','HOMO','LUMO',r'$\Delta G_{\mathrm{ads}}$', 'Adsorption energy']


def choose_font(requested=None):
    if requested:
        font_manager.findfont(requested, fallback_to_default=False)
        return requested
    for name in ('Arial','Nimbus Sans','Helvetica','DejaVu Sans'):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            pass
    raise RuntimeError('No suitable sans-serif font was found')


def style(font):
    plt.rcParams.update({
        'font.family':font,'font.size':9.0,'axes.titlesize':10.5,'axes.titleweight':'bold',
        'axes.labelsize':9.5,'axes.labelweight':'bold','axes.linewidth':1.35,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'xtick.direction':'in','ytick.direction':'in','xtick.major.width':1.1,
        'ytick.major.width':1.1,'xtick.major.size':4,'ytick.major.size':4,
        'axes.unicode_minus':False,'savefig.facecolor':'white','mathtext.fontset':'dejavusans',
    })


def axes_style(ax, grid='y'):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(pad=4)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid,color=GRID,lw=.55,zorder=0)


def title(ax, label, text):
    ax.set_title(f'({label})  {text}',loc='left',color=NAVY,pad=8)


def field_ticks(ax, visible=True):
    ax.set_xlim(-.35,12.35)
    ax.set_xticks(np.arange(13))
    if visible:
        ax.set_xticklabels(FIELD_LABELS,rotation=55,ha='right',rotation_mode='anchor')
        ax.tick_params(axis='x',pad=7)
    else:
        ax.tick_params(labelbottom=False)


def one(df, **filters):
    rows=df
    for key,value in filters.items():
        rows=rows[rows[key]==value]
    if len(rows)!=1:
        raise ValueError(f'Expected one row: {filters}, found {len(rows)}')
    return rows.iloc[0]


def field_values(df,model,strategy,metric):
    d=df[(df.model_key==model)&(df.evidence_strategy==strategy)].set_index('field')
    assert len(d)==13 and d.index.is_unique and set(d.index)==set(FIELDS)
    y=d.loc[FIELDS,metric].to_numpy(float)
    assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
    return y


def signed(x, places=3):
    return f'{x:+.{places}f}'


def figure5(data):
    d=data['e0_e5_strategy_metrics'];fig=plt.figure(figsize=(WIDTH,3.8))
    colors=['#6B6B6B','#E46C0A','#2F75B5','#299447','#D49A00','#7A4FB3']
    labels=['Field Micro F1','Record-matching F1','Null Accuracy']
    for i,(metric,label,left) in enumerate(zip(METRICS,labels,[.095,.405,.715])):
        ax=fig.add_axes([left,.185,.208,.62])
        rows=[one(d,strategy=f'E{j}',metric=metric) for j in range(6)]
        lows=[float(r.ci_low) for r in rows];highs=[float(r.ci_high) for r in rows]
        for j,(r,color) in enumerate(zip(rows,colors)):
            value=float(r['mean']);lo=float(r.ci_low);hi=float(r.ci_high)
            assert r.n_runs==9
            ax.errorbar(value,j,xerr=[[value-lo],[hi-value]],fmt='o',ms=7.5,
                color=color,mec='white',mew=.7,lw=2.5,capsize=4,capthick=1.8,zorder=4)
            ax.text(1.035,j,f'{value:.4f}',transform=ax.get_yaxis_transform(),
                ha='left',va='center',fontsize=9,color=NAVY,clip_on=False)
        ax.axvline(float(rows[0]['mean']),color='#818991',ls='--',lw=1.35,zorder=1)
        lower=max(0,np.floor((min(lows)-.008)/.02)*.02)
        upper=min(1,np.ceil((max(highs)+.008)/.02)*.02)
        ax.set_xlim(lower,upper);ax.set_ylim(5.65,-.65)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_yticks(range(6),[f'E{j}' for j in range(6)] if i==0 else ['']*6)
        ax.tick_params(axis='y',length=0)
        if i==0:
            for tick,color in zip(ax.get_yticklabels(),colors):
                tick.set_color(color);tick.set_fontweight('bold');tick.set_fontsize(10)
        ax.set_xlabel('Score',labelpad=6);axes_style(ax,grid='x')
        ax.set_title(f'({chr(97+i)}) {label}',loc='left',fontsize=10,color=NAVY,pad=12)
    return fig


def figureS1(data):
    d=data['e4_minus_e0_fields'].sort_values('mean').reset_index(drop=True)
    assert len(d)==13 and set(d.field)==set(FIELDS) and (d.n_runs==9).all()
    labels=dict(zip(FIELDS,FIELD_LABELS));values=d['mean'].to_numpy(float)
    fig=plt.figure(figsize=(WIDTH,5.2));ax=fig.add_axes([.33,.15,.625,.755])
    ax.barh(np.arange(13),values,height=.62,
        color=[TEAL if value>=0 else ORANGE for value in values],zorder=3)
    ax.axvline(0,color='#596273',lw=1.35,zorder=2)
    ax.set_yticks(np.arange(13),[labels[f] for f in d.field]);ax.tick_params(axis='y',length=0,pad=7)
    span=max(float(values.max()-values.min()),.05);offset=span*.025
    ax.set_xlim(min(0,float(values.min()))-.17*span,max(0,float(values.max()))+.17*span)
    ax.set_ylim(-.75,12.75)
    for j,value in enumerate(values):
        ax.text(value+(offset if value>=0 else -offset),j,f'{value:+.3f}',va='center',
            ha='left' if value>=0 else 'right',fontsize=9,color=NAVY)
    ax.set_xlabel('Field F1 difference (E4 − E0)',labelpad=8)
    ax.set_title('GLM-4-Plus',loc='left',color=NAVY,pad=12)
    axes_style(ax,grid='x')
    return fig


def figure6(data):
    d=data['test50_model_metrics']; paired=data['glm_paired_comparison']
    fig=plt.figure(figsize=(WIDTH,5.65))
    axs=[fig.add_axes([x,.55,.245,.29]) for x in (.075,.405,.735)]
    names=['Field Micro F1','Record-matching F1','Null Accuracy']
    handles=[]
    for k,(ax,metric,label) in enumerate(zip(axs,METRICS,names)):
        vals=np.array([[one(d,model_key=m,evidence_strategy=s)[metric] for s in STRATEGIES] for m in MODELS])
        for j in range(3):
            bars=ax.bar(np.arange(3)+(j-1)*.23,vals[:,j],width=.23,color=FACES[j],
                        edgecolor=EDGES[j],lw=1.0,zorder=3,label=f'Q{j}')
            if k==0: handles.append(bars[0])
        # Two tiers above each group: Q0/Q2 below, Q1 above.
        for i in range(3):
            for j in range(3):
                x=i+(j-1)*.23+[-.12,0,.12][j]
                y=vals[i].max()+(.09 if j==1 else .015)
                ax.text(x,y,f'{vals[i,j]:.3f}',ha='center',va='bottom',fontsize=8.4,
                        color=NAVY,fontweight='bold',clip_on=False)
        ax.set_xlim(-.65,2.65);ax.set_ylim(0,1);ax.set_yticks(np.arange(0,1.01,.2))
        ax.set_xticks(range(3),['3B','7B','14B'])
        ax.set_xlabel('Qwen2.5',labelpad=4)
        ax.set_ylabel(label,labelpad=4)
        ax.text(0,1.18,f'({chr(97+k)})',transform=ax.transAxes,fontweight='bold',fontsize=11,color=NAVY)
        axes_style(ax)
    fig.legend(handles,['Q0','Q1','Q2'],loc='upper center',bbox_to_anchor=(.535,.985),
               ncol=3,frameon=False,title='Evidence strategy',title_fontsize=9.5,
               handlelength=1.8,columnspacing=2.5)
    ax=fig.add_axes([.225,.13,.735,.265])
    names=['Field Micro F1','Record-matching F1','Null Accuracy']
    ax.axvline(0,color='#737D88',ls='--',lw=1.35)
    for i,m in enumerate(METRICS):
        row=one(paired,metric=m);y=2-i
        value,lo,hi=[float(row[x]) for x in ('difference_high_minus_low','ci_low','ci_high')]
        ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',color=TEAL,
                    markersize=7,lw=2.5,capsize=4,capthick=1.7,zorder=4)
        ax.text(.071,y,f'{signed(value)} [{signed(lo)}, {signed(hi)}]',fontsize=9,
                va='center',color=NAVY)
    ax.set_xlim(-.125,.15);ax.set_ylim(-.65,2.65);ax.set_yticks([2,1,0],names)
    ax.set_xticks([-.10,-.05,0,.05]);ax.set_xlabel('Difference (Qwen2.5-14B Q1 − GLM-4-Plus Q0)',labelpad=7)
    title(ax,'d','Paired difference with 95% CI')
    axes_style(ax,grid='x')
    return fig


def figure7(data):
    d=data['structure_differences']
    fig=plt.figure(figsize=(WIDTH,6.2))
    positions=[(.12,.555),(.615,.555),(.12,.15),(.615,.15)]
    specs=[('document_length_quartile','record_f1_difference','Document length'),
           ('table_density_quartile','record_f1_difference','Table density'),
           ('document_length_quartile','field_micro_f1_difference','Document length'),
           ('table_density_quartile','field_micro_f1_difference','Table density')]
    series=[('qwen2_5_3b','Q2-Q0','#2E73B8','o'),('qwen2_5_7b','Q2-Q0',TEAL,'s'),('qwen2_5_14b','Q1-Q0',ORANGE,'^')]
    strata=['Q1_low','Q2_mid_low','Q3_mid_high','Q4_high'];legend=[]
    displayed=[]
    for st,m,_ in specs:
        for model,comp,_,_ in series:
            displayed.extend(d[(d.stratum_type==st)&(d.model_key==model)&(d.comparison==comp)][m].to_numpy()*100)
    low=min(0,np.floor(min(displayed)/10)*10);high=max(10,np.ceil(max(displayed)/10)*10)
    for k,((left,bottom),(st,metric,axisname)) in enumerate(zip(positions,specs)):
        ax=fig.add_axes([left,bottom,.355,.27]);endpoints=[]
        for model,comp,color,marker in series:
            dd=d[(d.stratum_type==st)&(d.model_key==model)&(d.comparison==comp)].set_index('stratum')
            assert len(dd)==4
            vals=dd.loc[strata,metric].to_numpy()*100
            line,=ax.plot(range(1,5),vals,color=color,marker=marker,ms=5.8,lw=2.4)
            if k==0:legend.append(line)
            endpoints.append((vals[-1],color))
        # Offset endpoint labels only; observations themselves are never moved.
        ordered=sorted(range(3),key=lambda i:endpoints[i][0]);label_y={}
        minsep=(high-low)*.095
        for j,i in enumerate(ordered):
            label_y[i]=max(endpoints[i][0],label_y[ordered[j-1]]+minsep if j else -np.inf)
        excess=max(label_y.values())-(high-.04*(high-low))
        if excess>0:
            label_y={i:y-excess for i,y in label_y.items()}
        for i,(value,color) in enumerate(endpoints):
            ax.annotate(f'{value:+.1f}',xy=(4,value),xytext=(4.15,label_y[i]),
                color=color,fontsize=9,fontweight='bold',va='center',ha='left',
                arrowprops=dict(arrowstyle='-',color=color,lw=.8),annotation_clip=False)
        ax.axhline(0,color='#858E97',ls='--',lw=1.2,zorder=1)
        ax.set_ylim(low,high);ax.set_xlim(.85,4.83)
        ax.set_xticks(range(1,5),['1','2','3','4'])
        ax.set_yticks(np.arange(low,high+1,10))
        ax.set_xlabel(f'{axisname} quartile',labelpad=5)
        ax.set_ylabel(('Δ Record-matching F1 (pp)' if k<2 else 'Δ Field Micro F1 (pp)'),labelpad=5)
        title(ax,chr(97+k),axisname)
        axes_style(ax)
    fig.legend(legend,[f'{NAMES[m]}: {c.replace("-"," − ")}' for m,c,_,_ in series],
        loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,frameon=False,columnspacing=1.6,handlelength=2.3)
    return fig


def figure8(data):
    d=data['test50_field_metrics'];fig=plt.figure(figsize=(WIDTH,7.25))
    lines=[]
    for i,(metric,name,bottom) in enumerate(zip(['precision','recall','f1'],['Precision','Recall','F1'],[.705,.455,.205])):
        ax=fig.add_axes([.115,bottom,.855,.175])
        for j,(strategy,color,marker) in enumerate(zip(STRATEGIES[:2],[NAVY,TEAL],['o','s'])):
            y=field_values(d,'qwen2_5_14b',strategy,metric)
            line,=ax.plot(range(13),y,lw=2.4,marker=marker,ms=5.5,color=color)
            if i==0:lines.append(line)
        ax.set_ylim(0,1);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel(name,labelpad=7)
        field_ticks(ax,visible=i==2);axes_style(ax)
        title(ax,chr(97+i),name)
    fig.legend(lines,['Qwen2.5-14B Q0','Qwen2.5-14B Q1'],loc='upper center',
        bbox_to_anchor=(.54,.985),ncol=2,frameon=False,columnspacing=2,handlelength=2.5)
    return fig


def figureS2(data):
    d=data['dev37_prompt_matrix'];fig=plt.figure(figsize=(WIDTH,4.6))
    cmap=LinearSegmentedColormap.from_list('paper_heatmap',['#FAFCFD','#D9E8E6','#8FBDBD','#236D78','#14364F'])
    prompts=['recall','balanced','precision']
    for i,(metric,name,left) in enumerate(zip(METRICS,['Field Micro F1','Record-matching F1','Null Accuracy'],[.235,.485,.735])):
        ax=fig.add_axes([left,.275,.218,.585])
        values=np.array([[one(d,model_key=m,evidence_strategy=s,prompt_variant=p)[metric] for p in prompts]
                         for m in MODELS for s in STRATEGIES])
        im=ax.imshow(values,vmin=0,vmax=1,cmap=cmap,aspect='auto',interpolation='nearest')
        for row in range(9):
            for col in range(3):
                ax.text(col,row,f'{values[row,col]:.3f}',ha='center',va='center',fontsize=9,
                        color='white' if values[row,col]>.67 else NAVY)
        ax.set_yticks(range(9),[f'{NAMES[m]}  Q{j}' for m in MODELS for j in range(3)] if i==0 else ['']*9)
        ax.set_xticks(range(3),['Recall','Balanced','Precision'],rotation=35,ha='right',rotation_mode='anchor')
        ax.tick_params(length=0,pad=6)
        for spine in ax.spines.values():spine.set_visible(False)
        for y in [2.5,5.5]:ax.axhline(y,color='white',lw=2)
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,9,1),minor=True)
        ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='minor',length=0)
        ax.text(0,1.09,f'({chr(97+i)})',transform=ax.transAxes,fontweight='bold',fontsize=11,color=NAVY)
        ax.set_title(name,fontdict={'size':9.4,'weight':'bold'},pad=5,color=NAVY)
    cax=fig.add_axes([.345,.095,.52,.03]);cbar=fig.colorbar(im,cax=cax,orientation='horizontal')
    cbar.set_ticks([0,.2,.4,.6,.8,1]);cbar.set_label('Score',fontsize=9.5,labelpad=3)
    cbar.outline.set_linewidth(.8);cbar.ax.tick_params(length=3)
    return fig


def figureS3(data):
    d=data['test50_paired_comparisons'];fig=plt.figure(figsize=(WIDTH,6.15))
    blocks=[('strategy',[('qwen2_5_3b','Q2-Q0','Qwen2.5-3B\nQ2 − Q0'),
                        ('qwen2_5_7b','Q2-Q0','Qwen2.5-7B\nQ2 − Q0'),
                        ('qwen2_5_14b','Q1-Q0','Qwen2.5-14B\nQ1 − Q0')],.565,.285,'Strategy contrasts'),
            ('model',[('q0_safe_full','7B-3B','Qwen2.5-7B −\nQwen2.5-3B (Q0)'),
                      ('q0_safe_full','14B-7B','Qwen2.5-14B −\nQwen2.5-7B (Q0)')],.13,.235,'Capacity contrasts')]
    handles=[Line2D([],[],marker='o',color=NAVY,lw=2.5,ms=7),Line2D([],[],marker='s',color=TEAL,lw=2.5,ms=7)]
    fig.legend(handles,['Field Micro F1','Record-matching F1'],loc='upper center',bbox_to_anchor=(.54,.995),
               ncol=2,frameon=False,columnspacing=1.6,handlelength=2)
    for k,(family,groups,bottom,height,heading) in enumerate(blocks):
        ax=fig.add_axes([.295,bottom,.355,height]);centers=np.arange(len(groups))[::-1]*1.3
        for center,(model,comp,_) in zip(centers,groups):
            for j,(metric,color,marker) in enumerate(zip(METRICS[:2],[NAVY,TEAL],['o','s'])):
                row=one(d,comparison_family=family,comparison=comp,model_or_strategy=model,metric=metric)
                value,lo,hi=[float(row[x]) for x in ('difference_high_minus_low','ci_low','ci_high')]
                y=center+(.23 if j==0 else -.23)
                ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt=marker,color=color,
                    markersize=7,lw=2.7,capsize=4,capthick=1.7,zorder=3)
                ax.text(1.06,y,f'{signed(value)} [{signed(lo)}, {signed(hi)}]',
                    transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=9,color=color,clip_on=False)
        ax.set_ylim(-.7,centers.max()+.7);ax.set_xlim(-.04,.32)
        ax.set_xticks([0,.1,.2,.3]);ax.set_yticks(centers,[x[2] for x in groups]);ax.tick_params(axis='y',length=0,pad=9)
        ax.axvline(0,color='#858E97',ls='--',lw=1.3)
        ax.set_xlabel('Difference in F1',labelpad=6);axes_style(ax,grid='x')
        fig.text(.035,bottom+height+.05,f'({chr(97+k)})  {heading}',fontsize=10.5,fontweight='bold',color=NAVY)
        ax.text(1.06,1.05,'Difference [95% CI]',transform=ax.transAxes,fontweight='bold',fontsize=9,color=NAVY)
    return fig


def figureS4(data):
    d=data['test50_field_metrics'];fig=plt.figure(figsize=(WIDTH,6.8))
    ax=fig.add_axes([.115,.59,.855,.255]);bx=fig.add_axes([.115,.22,.855,.235]);handles=[]
    series=[('qwen2_5_14b','q0_safe_full',NAVY,'o'),('qwen2_5_14b','q1_budgeted_rag',TEAL,'s'),
            ('qwen2_5_14b','q2_table_aware_rag',ORANGE,'^'),('glm_4_plus','q0_safe_full','#8B6EA6','D')]
    for model,strategy,color,marker in series:
        y=field_values(d,model,strategy,'f1');line,=ax.plot(range(13),y,color=color,marker=marker,ms=5.2,lw=2.2)
        handles.append(line)
    q0=field_values(d,'qwen2_5_14b',STRATEGIES[0],'f1');diffs=[]
    for s,color,marker in zip(STRATEGIES[1:],[TEAL,ORANGE],['s','^']):
        y=(field_values(d,'qwen2_5_14b',s,'f1')-q0)*100;diffs.extend(y)
        bx.plot(range(13),y,color=color,marker=marker,ms=5.5,lw=2.4,
                label='Q1 − Q0' if s==STRATEGIES[1] else 'Q2 − Q0')
    ax.set_ylim(0,1);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('Field F1',labelpad=7)
    low=min(-5,np.floor(min(diffs)/5)*5);high=max(5,np.ceil(max(diffs)/5)*5)
    bx.set_ylim(low,high);bx.set_ylabel('Δ Field F1 (pp)',labelpad=7);bx.axhline(0,color='#858E97',ls='--',lw=1.3)
    field_ticks(ax,False);field_ticks(bx,True);axes_style(ax);axes_style(bx)
    title(ax,'a','Field-level comparison');title(bx,'b','Qwen2.5-14B: difference from Q0')
    fig.legend(handles,['Qwen2.5-14B Q0','Qwen2.5-14B Q1','Qwen2.5-14B Q2','GLM-4-Plus Q0'],
        loc='upper center',bbox_to_anchor=(.54,.995),ncol=2,frameon=False,columnspacing=1.7,handlelength=2.3)
    bx.legend(loc='upper left',ncol=2,frameon=False,fontsize=9,columnspacing=1.3)
    return fig


FIGURES = {
    '5':('Figure_5_E0_E5_performance_t038',figure5),
    '6':('Figure_6_capacity_strategy_t038',figure6),
    '7':('Figure_7_document_structure_t038',figure7),
    '8':('Figure_8_field_precision_recall_f1_t038',figure8),
    'S1':('Figure_S1_E4_minus_E0_fields_t038',figureS1),
    'S2':('Figure_S2_dev37_prompt_matrix_t038',figureS2),
    'S3':('Figure_S3_paired_comparisons_t038',figureS3),
    'S4':('Figure_S4_field_comparisons_t038',figureS4),
}


def save(fig, name, output, font):
    tif=output/'TIFF'/f'{name}.tif';png=output/'preview'/f'{name}.png'
    tif.parent.mkdir(parents=True,exist_ok=True);png.parent.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    outside=[]
    for txt in fig.findobj(match=Text):
        if not txt.get_visible() or not txt.get_text().strip():continue
        box=txt.get_window_extent(renderer)
        if box.width==0 or box.height==0:continue
        if box.x0<bounds.x0-1 or box.x1>bounds.x1+1 or box.y0<bounds.y0-1 or box.y1>bounds.y1+1:
            outside.append(txt.get_text())
    if outside:
        raise RuntimeError(f'Text outside figure {name}: {outside}')
    # Render once, then publish only a fully decoded and verified TIFF.
    # S3 uses baseline uncompressed TIFF for maximum reader compatibility.
    compression='raw' if name==FIGURES['S3'][0] else 'tiff_lzw'
    compression_tag=1 if compression=='raw' else 5
    buffer=BytesIO()
    fig.savefig(buffer,dpi=600,format='png',facecolor='white')
    buffer.seek(0)
    with Image.open(buffer) as im:
        rgba=im.convert('RGBA');white=Image.new('RGBA',rgba.size,'white')
        rgb=Image.alpha_composite(white,rgba).convert('RGB')
    buffer.close()
    pending=tif.with_name(tif.stem+'.writing.tif')
    rgb.save(pending,format='TIFF',compression=compression,dpi=(600,600))
    with Image.open(pending) as im:
        im.load()
        assert im.mode=='RGB' and im.tag_v2[259]==compression_tag
        assert all(abs(x-600)<.01 for x in im.info['dpi'])
        assert im.tobytes()==rgb.tobytes(), 'TIFF pixel round-trip failed'
        result=dict(file=str(tif.relative_to(output)),pixels=list(im.size),dpi=[float(x) for x in im.info['dpi']],
                    mode=im.mode,compression='uncompressed' if compression_tag==1 else 'LZW',
                    font_family=font,all_text_inside_canvas=True,pixel_round_trip_verified=True,
                    sha256=hashlib.sha256(pending.read_bytes()).hexdigest())
    pending.replace(tif)
    pending_png=png.with_name(png.stem+'.writing.png')
    fig.savefig(pending_png,dpi=300,format='png',facecolor='white')
    with Image.open(pending_png) as im:im.load()
    pending_png.replace(png)
    plt.close(fig);print(f'Saved {tif.name}',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    here=Path(__file__).resolve().parent
    p.add_argument('--data-dir',type=Path,default=here/'data')
    p.add_argument('--output-dir',type=Path,default=here)
    p.add_argument('--figures',nargs='+',choices=list(FIGURES),default=list(FIGURES))
    p.add_argument('--font',help='Optional installed family, e.g. Arial; fail if unavailable')
    a=p.parse_args();font=choose_font(a.font);style(font)
    prov=json.loads((a.data_dir/'provenance.json').read_text())
    assert prov['record_threshold']==.38 and prov['verified_qwen_runs']==48 and prov['glm_test_papers']==50
    if any(key in a.figures for key in ['5','S1']):
        assert prov.get('verified_e0_e5_runs')==54, 'Verified E0-E5 data are required'
    data={}
    for filename,digest in prov['plot_data_sha256'].items():
        path=a.data_dir/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, f'Data changed: {filename}'
        data[path.stem]=pd.read_csv(path)
    qa_path=a.output_dir/'figure_quality_checks.json'
    previous=json.loads(qa_path.read_text()) if qa_path.exists() else {}
    reports={row['file']:row for row in previous.get('figures',[])}
    for key in a.figures:
        name,func=FIGURES[key];report=save(func(data),name,a.output_dir,font)
        reports[report['file']]=report
    qa_path.write_text(json.dumps(dict(
        matplotlib_version=matplotlib.__version__,record_threshold=.38,figures=list(reports.values())),indent=2)+'\n')


if __name__=='__main__':
    main()
