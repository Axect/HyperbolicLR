import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import scienceplots
import numpy as np

# Prepare Data to Plot
N = 10
U = 15

def h(n):
    return np.sqrt((N-n) / U * (2 - (N+n) / U))

def asymptote(n):
    return -1/U * (n - U)

x = np.linspace(0, N, 1000)
y = h(x)

x_u = np.linspace(-2, U+2, 1000)
z = asymptote(x_u)

# Plot params
pparam = dict(
    xscale = 'linear',
    yscale = 'linear',
    xlim   = (-2, U+2),
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x, y, '-', color='darkblue', label=r'$h(n;N,U)$')
    ax.plot(x_u, z, ':', color='darkblue', label=r'Asymptote', alpha=0.5)
    ax.legend()

    # Remove all ticks
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # Add custom text
    ax.text(10, -0.06, r'$N$', ha='center')
    ax.text(15, -0.06, r'$U$', ha='center')
    ax.text(-0.4, 1, r'$1$', va='center')

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add arrows to the end of the axes
    arrowprops = dict(arrowstyle='->', linewidth=1.5, color='black')
    ax.annotate('', xy=(U+2, 0), xytext=(-2, 0), arrowprops=arrowprops)
    ax.annotate('', xy=(0, 1+2/15), xytext=(0, -2/15), arrowprops=arrowprops)
    ax.text(16, 0.03, r'$n$', ha='center')
    ax.text(0.3, 1+1/15, r'$h$', va='center')

    fig.savefig('../figs/02_prop_1.png', dpi=600, bbox_inches='tight')
