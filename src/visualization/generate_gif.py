import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from simulation.physics_model import CarParams, TrackParams, run_lap

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
BG='#0A0A0C'; P1='#13131A'; GRN='#00E678'; YEL='#FFD732'; RED='#D21E1E'; WHT='#F0F0F5'; GRY='#787880'

def make_gif(res, path, step=40, fps=12):
    s=res['s']; T=res['T_tire']; mu=res['mu_eff']; SOC=res['SOC']*100; N=len(s)
    frames=list(range(0,N,step))
    fig,axes=plt.subplots(3,1,figsize=(10,8),dpi=80)
    fig.patch.set_facecolor(BG)
    for ax in axes: ax.set_facecolor(P1)
    lines=[ax.plot([],[],lw=1.5)[0] for ax in axes]
    axes[0].set_xlim(s[0],s[-1]); axes[0].set_ylim(35,155)
    axes[1].set_xlim(s[0],s[-1]); axes[1].set_ylim(0.35,2.25)
    axes[2].set_xlim(s[0],s[-1]); axes[2].set_ylim(15,100)
    axes[0].set_ylabel('Tyre Temp C',color=YEL)
    axes[1].set_ylabel('mu_eff',color=GRN)
    axes[2].set_ylabel('SOC pct',color=GRN)
    axes[2].set_xlabel('Arc length m')
    axes[0].axhline(95,color=RED,lw=0.7,ls='--',alpha=0.6)
    axes[2].axhline(35,color=RED,lw=0.7,ls='--',alpha=0.6)
    for ax in axes: ax.grid(True,color='#222230',lw=0.4)
    title=fig.suptitle('',color=WHT,fontsize=10,fontweight='bold')
    def init():
        for ln in lines: ln.set_data([],[])
        return lines+[title]
    def update(fi):
        i=frames[fi]
        lines[0].set_data(s[:i+1],T[:i+1]); lines[0].set_color(YEL)
        lines[1].set_data(s[:i+1],mu[:i+1]); lines[1].set_color(GRN)
        lines[2].set_data(s[:i+1],SOC[:i+1]); lines[2].set_color(GRN)
        pct=i/N*100
        title.set_text('F1 Lap Sim  %.0f pct  T=%.1fC  mu=%.3f  SOC=%.1f pct' % (pct,T[i],mu[i],SOC[i]))
        return lines+[title]
    ani=animation.FuncAnimation(fig,update,frames=len(frames),init_func=init,blit=True,interval=1000//fps)
    ani.save(path,writer='pillow',fps=fps)
    plt.close(fig)
    print('GIF saved: '+path)

if __name__=='__main__':
    cp=CarParams(); tp=TrackParams()
    print('Running simulation...')
    res=run_lap(cp,tp)
    print('Lap time: %.2f s' % res['lap_time'])
    make_gif(res,os.path.join(OUT_DIR,'tyre_thermal_evolution.gif'))
