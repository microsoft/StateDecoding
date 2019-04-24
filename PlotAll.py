import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import pickle
import Params
import matplotlib.ticker as mticker
import matplotlib.transforms
import matplotlib.patches as mpatches

FONTSIZE=14
AXISFONT=12

colors = {
    'decoding': '#000000',
    'decoding_0.1': '#e41a1c',
    'decoding_0.2': '#984ea3',
    'decoding_0.3': '#ff7f00',
    'oracleq': '#377eb8',
    'qlearning': '#4daf4a',
}

linestyles = {
    'decoding': '-',
    'decoding_0.1': '-',
    'decoding_0.2': '-',
    'decoding_0.3': '-',
    'oracleq': '--',
    'qlearning': '--',
}

markertypes = {
    'linear': 'o',
    'nn': 's'
}

Yaxlimits = {
    'solve': [300, 100000],
    'find': [100, 100000],
}
Xaxlimits = {
    'solve': [5,50],
    'find': [5,50],
}
PrettyTitle = {
    'Lock-v0': 'Lock-Bernoulli',
    'Lock-v1': 'Lock-Gaussian',
    'Lock-v2': 'Lock-Logistic',
}

def parse_args():
    parser = argparse.ArgumentParser(description='StateDecoding Plotting Script')
    parser.add_argument('--loglog', type=bool, default=False,
                        help='LogLog or SemiLogY')
    args = parser.parse_args()
    return(args)

def set_axes_info(axes,prefix):
    for ax in axes:
        ax.set_ylim(Yaxlimits[prefix])
        ax.set_xlim(Xaxlimits[prefix])
        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks([5,10,20,50], minor=False)
        ax.set_xticklabels(["5","10","20","50"], minor=False)
        ax.set_xticklabels(["","","",""],minor=True)
        ax.yaxis.set_ticks_position('both')
        dx = 0/72.; dy=-2/72.
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
        dx = 2/72.; dy=0/72.
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        for label in ax.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

    axes[1].xaxis.tick_top()
    axes[1].set_xticklabels([])
    axes[3].xaxis.tick_top()
    axes[3].set_xticklabels([])
    axes[5].xaxis.tick_top()
    axes[5].set_xticklabels([])

    axes[0].set_yticklabels([])
    axes[1].set_yticklabels([])


def load_data(env,prefix,ep1,model_type='linear'):
    algs = ['oracleq', 'qlearning', 'decoding']
    if env=='Lock-v1' or env=='Lock-v2':
        algs.remove('decoding')
        algs.extend(['decoding_0.1', 'decoding_0.2', 'decoding_0.3'])

    Data = {}
    for alg in algs:
        if alg[0:4] != 'deco':
            (arr, find_best, solve_best) = pickle.load(open("./pkls/%s_%s_%s_%s_%s.pkl" % ('Lock-v0',alg,'linear',ep1,"None"), "rb"))
            Data[alg] = arr
        else:
            if alg == 'decoding':
                (name,param) = ('decoding', 'None')
            else:
                (name,param) = alg.split("_")
            (arr,find_best,solve_best) = pickle.load(open("./pkls/%s_%s_%s_%s_%s.pkl" % (env,name,model_type,ep1,param), "rb"))
            Data[alg] = arr
    return Data

def plot_curves(ax,Data,args,model_type='linear',prefix='solve'):
    xpoints = Params.LockHorizons['decoding']
    for k in Data.keys():
        print(k)
        inds = np.where(np.array(Data[k][prefix]) == Params.T+100)
        if len(inds[0]) > 0:
            ind = inds[0][0]
            ypoints = Data[k][prefix][0:ind]
            ypoints.extend([Params.T for i in range(len(xpoints) - len(ypoints))])
            if args.loglog:
                l1 = ax.loglog(Data[k]['horizons'][0:ind], Data[k][prefix][0:ind], linewidth=2, linestyle=linestyles[k], markersize=5, marker='o', color=colors[k])
            else:
                l1 = ax.semilogy(Data[k]['horizons'][0:ind], Data[k][prefix][0:ind], linewidth=2, linestyle=linestyles[k], markersize=5, marker='o', color=colors[k])
            if k[0:4] == 'deco':
                l1[0].set_marker(markertypes[model_type])
            ax.fill_between(Data[k]['horizons'][0:ind], Data[k][prefix+'_low'][0:ind], Data[k][prefix+'_high'][0:ind], alpha=0.3,color=colors[k])
            if k[0:4] == 'deco':
                if args.loglog:
                    l1 = ax.loglog([Data[k]['horizons'][ind-1]], [Data[k][prefix][ind-1]], marker='o', markersize=10, color=colors[k], label='_nolegend_')
                else:
                    l1 = ax.semilogy([Data[k]['horizons'][ind-1]], [Data[k][prefix][ind-1]], marker='o', markersize=10, color=colors[k], label='_nolegend_')
                l1[0].set_marker(markertypes[model_type])
        else:
            if args.loglog:
                l1 = ax.loglog(Data[k]['horizons'], Data[k][prefix], linewidth=2, linestyle=linestyles[k], markersize=5, marker='o', color=colors[k])
            else:
                l1 = ax.semilogy(Data[k]['horizons'], Data[k][prefix], linewidth=2, linestyle=linestyles[k]
, markersize=5, marker='o', color=colors[k])
            if k[0:4] == 'deco':
                l1[0].set_marker(markertypes[model_type])
            ax.fill_between(Data[k]['horizons'], Data[k][prefix+'_low'], Data[k][prefix+'_high'], alpha=0.3,color=colors[k])
            
if __name__=='__main__':
    args = parse_args()
    print(args, flush=True)

    plt.rc('font',size=AXISFONT)
    plt.rc('font', family='sans-serif')

    for prefix in ['solve', 'find']:
        fig = plt.figure(figsize=(mpl.rcParams['figure.figsize'][0]*2.6, mpl.rcParams['figure.figsize'][1]*1.25))
        ax = fig.add_subplot(111,frameon=False)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        
        Data = load_data('Lock-v0', prefix, '0.0')
        ax1 = fig.add_subplot(231)
        ax1.grid(linestyle='--')
        plot_curves(ax1,Data,args,prefix=prefix)

        Data = load_data('Lock-v0', prefix, '0.1')
        ax2 = fig.add_subplot(234)
        ax2.grid(linestyle='--')
        plot_curves(ax2,Data,args,prefix=prefix)

        Data = load_data('Lock-v1', prefix, '0.0')
        ax3 = fig.add_subplot(232)
        ax3.grid(linestyle='--')
        plot_curves(ax3,Data,args,prefix=prefix)

        Data = load_data('Lock-v1', prefix, '0.1')
        ax4 = fig.add_subplot(235)
        ax4.grid(linestyle='--')
        plot_curves(ax4,Data,args,prefix=prefix)

        Data = load_data('Lock-v1', prefix, '0.0',model_type='nn')
        ax5 = fig.add_subplot(233)
        ax5.grid(linestyle='--')
        plot_curves(ax5,Data,args,model_type='nn',prefix=prefix)

        Data = load_data('Lock-v1', prefix, '0.1',model_type='nn')
        ax6 = fig.add_subplot(236)
        ax6.grid(linestyle='--')
        plot_curves(ax6,Data,args,model_type='nn',prefix=prefix)
        ax.set_xlabel('Horizon = context dimension',labelpad=-10,fontsize=FONTSIZE)
        ax.set_ylabel('Time to ' + prefix,fontsize=FONTSIZE)
        

        from matplotlib.lines import Line2D
        oracleq = Line2D(['0'],['0'], color = colors['oracleq'], linestyle=linestyles['oracleq'], markersize=5, marker=markertypes['linear'], label='OracleQ (No noise)')
        qlearning = Line2D(['0'],['0'], color = colors['qlearning'], linestyle=linestyles['oracleq'], markersize=5, marker=markertypes['linear'], label='QLearning (No noise)')
        decoding = Line2D(['0'],['0'], color = colors['decoding'], linestyle=linestyles['decoding'], markersize=5, marker=markertypes['linear'], label='PCID:Lin (Ber noise)')
        deco_1_linear = Line2D(['0'],['0'], color = colors['decoding_0.1'], markersize=5, marker=markertypes['linear'], label='PCID:Lin (Gauss $\sigma=0.1$)')
        deco_1_nn = Line2D(['0'],['0'], color = colors['decoding_0.1'], markersize=5, marker=markertypes['nn'], label='PCID:NN (Gauss $\sigma=0.1$)')
        deco_2_linear = Line2D(['0'],['0'], color = colors['decoding_0.2'], markersize=5, marker=markertypes['linear'], label='PCID:Lin (Gauss $\sigma=0.2$)')
        deco_2_nn = Line2D(['0'],['0'], color = colors['decoding_0.2'], markersize=5, marker=markertypes['nn'], label='PCID:NN (Gauss $\sigma=0.2$)')
        deco_3_linear = Line2D(['0'],['0'], color = colors['decoding_0.3'], markersize=5, marker=markertypes['linear'], label='PCID:Lin (Gauss $\sigma=0.3$)')
        deco_3_nn = Line2D(['0'],['0'], color = colors['decoding_0.3'], markersize=5, marker=markertypes['nn'], label='PCID:NN (Gauss $\sigma=0.3$)')

        p1 = mpatches.Patch(color=colors['oracleq'], label='OracleQ')
        p2 = mpatches.Patch(color=colors['qlearning'], label='QLearning')
        p3 = mpatches.Patch(color=colors['decoding'], label='PCID:Lin')
        p4 = mpatches.Patch(color=colors['decoding_0.1'], label='PCID (obs noise 0.1)')
        p5 = mpatches.Patch(color=colors['decoding_0.2'], label='PCID (obs noise 0.2)')
        p6 = mpatches.Patch(color=colors['decoding_0.3'], label='PCID (obs noise 0.3)')
        ep = mpatches.Patch(color='white', label='',alpha=0.0)
#         leg = ax4.legend(handles=[p1,p2,p3,p4,p5,p6,ep,ep,ep,deco_1_linear,deco_1_nn], loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=False, ncol=6, frameon=False,fontsize=12)
#         leg1 = ax1.legend(handles=[oracleq,qlearning,decoding], loc='lower right', bbox_to_anchor=(0.9,-0.1))
#         leg2 = ax3.legend(handles=[oracleq,qlearning,deco_1_linear,deco_2_linear,deco_3_linear], loc='lower right', bbox_to_anchor=(0.95,-0.1))
#         leg3 = ax5.legend(handles=[oracleq,qlearning,deco_1_nn,deco_2_nn,deco_3_nn], loc='lower right', bbox_to_anchor=(0.95,-0.1))

        leg = ax4.legend(handles=
                         [oracleq,ep,qlearning,ep,decoding,ep,deco_1_linear,deco_1_nn,deco_2_linear,deco_2_nn,deco_3_linear,deco_3_nn], 
                         loc='upper center', bbox_to_anchor=(0.45, -0.12), fancybox=False, shadow=False, ncol=6, frameon=False,fontsize=FONTSIZE,
                         handletextpad=0.2,columnspacing=1)
#         if args.loglog:
#             ax4.text(0.7,30,'Note: Larger marker denotes next point is off the plot.')
#         else:
#             ax4.text(-33,30,'Note: Larger marker denotes next point is off the plot.')
#         leg = ax4.legend([
#                 'OracleQ',
#                 'QLearning',
#                 'Deco:Lin',
#                 '(center) Deco:Lin (obs noise 0.1)\n(right) Deco:NN (obs noise 0.1)',
#                 '(center) Deco:Lin (obs noise 0.2)\n(right) Deco:NN (obs noise 0.2)',
#                 '(center) Deco:Lin (obs noise 0.3)\n(right) Deco:NN (obs noise 0.3)'], loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=False, ncol=7, frameon=False,fontsize=12)
#         keys=['oracleq', 'qlearning', 'decoding', 'decoding_0.1', 'decoding_0.2', 'decoding_0.3']
#         i = 0
#         for legobj in leg.legendHandles:
#             legobj.set_linewidth(6.0)
#             legobj.set_color(colors[keys[i]])
#             legobj.set_markercolor(colors[keys[i]])
#             i += 1

        axes = [ax1,ax2,ax3,ax4,ax5,ax6]
        set_axes_info(axes,prefix)
        
        plt.sca(ax1)
        plt.title('Lock-Bernoulli (PCID uses linear model)',fontsize=FONTSIZE)
        plt.sca(ax3)
        plt.title('Lock-Gaussian (PCID uses linear model)',fontsize=FONTSIZE)
        plt.sca(ax5)
        plt.title('Lock-Gaussian (PCID uses NN model)',fontsize=FONTSIZE)
        
        ax1.set_ylabel('Deterministic')
        ax2.set_ylabel('Stochastic')

        plt.tight_layout(pad=0.1,w_pad=0.1,h_pad=0.1)
        plt.subplots_adjust(wspace=0.1,bottom=0.25)
        plt.savefig('./figs/plots_%s_%s.pdf' % (prefix, 'loglog' if args.loglog else 'semilog'), format='pdf', dpi=100, bbox_inches='tight')
        plt.close()

