import matplotlib.pyplot as plt
import scienceplots
import numpy as np

# Polynomial LR
def poly_lr(max_iter, power, init_lr):
    return lambda x: init_lr * (1 - x / max_iter) ** power

# Cosine Annealing LR
def cos_lr(max_iter, eta_min, init_lr):
    return lambda x: eta_min + (init_lr - eta_min) * (np.cos(np.pi * x / max_iter) + 1) / 2

N_vec = np.array([250, 500, 750, 1000])
epoch_vec = [np.arange(N) for N in N_vec]
eta_init = 1e-2
eta_min = 1e-6
eta_vec_poly = [poly_lr(N, 0.5, eta_init)(x) for x, N in zip(epoch_vec, N_vec)]
eta_vec_cos  = [cos_lr(N, eta_min, eta_init)(x) for x, N in zip(epoch_vec, N_vec)]

# Plot params
pparam = dict(
    xlabel = r'Epoch',
    ylabel = r'$\eta$',
    xscale = 'linear',
    yscale = 'linear',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    for i in range(len(N_vec)):
        ax.plot(epoch_vec[i], eta_vec_poly[i], label=f'N = {N_vec[i]}')
    ax.legend()
    fig.savefig('01_poly_lr.png', dpi=600, bbox_inches='tight')

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    for i in range(len(N_vec)):
        ax.plot(epoch_vec[i], eta_vec_cos[i], label=f'N = {N_vec[i]}')
    ax.legend()
    fig.savefig('01_cos_lr.png', dpi=600, bbox_inches='tight')
