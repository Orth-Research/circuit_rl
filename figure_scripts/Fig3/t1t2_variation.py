import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

noiseless_data = [0.600, 0.771, 0.864, 0.904, 0.925, 0.967]
noisy_data = [0.802, 0.924, 0.953, 0.964, 0.970, 0.980]
times = [1, 5, 10, 15, 20, 50]

f, (ax, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [4, 1]}, figsize=(5,5))
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

ax.scatter(times, noiseless_data, label="Noiseless MDP")
ax.scatter(times, noisy_data, label="Noisy MDP")

ax2.scatter([100], [0.998], label="Noiseless MDP")
ax2.scatter([100], [0.998], label="Noisy MDP", zorder=0)

ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.tick_right()

# ax2.set_xticklabels(['', '',r"$\infty$"])

positions = [100]
labels = [r'$\infty$']
ax2.xaxis.set_major_locator(ticker.FixedLocator(positions))
ax2.xaxis.set_major_formatter(ticker.FixedFormatter(labels))


for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.3)
    ax2.spines[axis].set_linewidth(1.3)

# ax2.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
ax2.get_xticklabels()[-1].set_fontsize(16)

d = .025  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

ax.annotate('HTTTTTHTHTHTHTTTH', (times[0]+0.2, noiseless_data[0]+0.003))
ax.annotate('HTHTTTTTTH', (times[0]+0.2, noisy_data[0]+0.003))
ax.annotate('HTHTHTH', (times[1]-3.4, noisy_data[1]+0.003))

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-4*d, +4*d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((- 4*d, + 4*d), (- d, + d), **kwargs)  # bottom-right diagonal
ax.set_ylabel("$\mathcal{F}$", fontsize=16)
ax.set_xlabel("T1, T2 times ($\mu s$)", fontsize=16)
ax.set_xscale('log')


handles,labels = ax.get_legend_handles_labels()
order = [1,0]
ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right', fontsize=12)
# ax2.legend(loc='lower right', fontsize=12)

# plt.savefig('../../figures/t1t2_variation', dpi=1000, transparent=False, bbox_inches='tight')
