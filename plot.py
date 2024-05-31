import matplotlib.pyplot as plt
import scienceplots
import numpy as np

# Prepare Data to Plot
U = 1000
N_vec = np.array([250, 500, 750, 995])
eta_init = 1e-2
eta_infimum = 1e-6
delta_eta = eta_init - eta_infimum
epoch_vec = [np.arange(N) for N in N_vec]
eta_vec = [
    eta_init + delta_eta * (np.sqrt((N - x) / U * (2 - (N + x) / U)) - np.sqrt(N / U * (2 - N / U)))
    for x, N in zip(epoch_vec, N_vec)
]
eta_grad_vec = delta_eta / np.sqrt(N_vec / U * (2 - N_vec / U)) * (- 2 / U)

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
        ax.plot(epoch_vec[i], eta_vec[i], label=f'N = {N_vec[i]}')
    ax.legend()
    fig.savefig('plot_eta.png', dpi=600, bbox_inches='tight')

# Plot params
pparam = dict(
    xlabel = r'N',
    ylabel = r'$\eta^\prime(0)$',
    xscale = 'linear',
    yscale = 'linear',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(N_vec, eta_grad_vec)
    ax.legend()
    fig.savefig('plot_grad.png', dpi=600, bbox_inches='tight')
