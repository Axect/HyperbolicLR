import matplotlib.pyplot as plt
import scienceplots
import numpy as np

# Prepare Data to Plot
epochs = [50, 100, 150, 200]
models = ['TraONet', 'DeepONet', 'LSTM', 'CNN', 'ResNet', 'ViT']

# Scheduler groups for subplot organization
group_A = ['N', 'L', 'S', 'P', 'C', 'E']       # Traditional
group_B = ['OC', 'CY', 'WC']                     # Cyclic / Warm-up
group_C = ['H', 'EH', 'WH', 'WEH']              # Hyperbolic
group_D = ['Prodigy', 'DAdapt']                   # LR-free

all_schedulers = group_A + group_B + group_C + group_D

# ── Existing data (unchanged) ────────────────────────────────────────────────
# These are the original results for the 6 original schedulers [N, P, C, E, H, EH].
# New scheduler data will be appended after experiments are run.

# Validation loss
TraONet_loss    = [
    [5.0977E-05,4.1721E-05,2.2007E-05,1.0926E-05],
    [2.8082E-05,8.9639E-06,5.2343E-06,4.0906E-06],
    [3.4426E-05,9.2909E-06,5.2014E-06,4.1982E-06],
    [3.5151E-05,1.6451E-05,1.2681E-05,1.1541E-05],
    [3.1015E-05,8.9335E-06,2.8333E-06,1.3565E-06],
    [2.9069E-05,9.9736E-06,4.9011E-06,2.8939E-06],
]
# SLCD (Smoothed Learning Curve Difference)
TraONet_SLCD = [
    [],
    [1.07E-01,1.39E-01,1.42E-01,1.49E-01,1.95E-01,1.72E-01],
    [1.25E-01,2.11E-01,2.10E-01,2.58E-01,3.41E-01,2.42E-01],
    [],
    [8.47E-02,7.90E-02,9.58E-02,1.29E-01,1.66E-01,1.50E-01],
    [1.89E-02,2.29E-02,4.28E-02,5.50E-02,6.69E-02,6.12E-02],
]

# Validation loss
DeepONet_loss = [
    [8.7109E-05, 1.2621E-04, 4.1373E-05, 3.0430E-05],
    [4.8261E-05, 6.7622E-04, 4.0187E-04, 1.7878E-04],
    [3.9565E-05, 3.1990E-04, 6.1428E-04, 1.3581E-02],
    [5.9143E-05, 3.8843E-05, 3.0233E-05, 2.9573E-05],
    [2.7446E-05, 1.9505E-03, 1.4471E-03, 4.5085E-04],
    [6.2508E-05, 5.0815E-05, 2.6157E-05, 2.1123E-05],
]
DeepONet_SLCD = [
    [],
    [5.59E-01, 4.12E-01, 6.36E-01, 2.42E-01, 4.39E-01, 2.92E-01],
    [5.88E-01, 5.20E-01, 5.58E-01, 6.09E-01, 7.56E-01, 7.24E-01],
    [],
    [3.35E-01, 4.85E-01, 5.24E-01, 5.25E-01, 5.26E-01, 3.00E-01],
    [5.08E-02, 5.51E-02, 4.92E-02, 4.34E-02, 6.85E-02, 4.58E-02],
]

# Validation loss
LSTM_loss = [
    [4.2841E-06, 3.2871E-06, 8.1370E-06, 7.1973E-06],
    [8.8830E-08, 4.2930E-08, 1.2091E-07, 1.5259E-08],
    [9.5646E-08, 3.3486E-08, 4.1540E-08, 2.0891E-08],
    [1.0610E-07, 6.1057E-08, 5.4890E-08, 5.4365E-08],
    [8.8849E-07, 7.1282E-08, 3.3126E-08, 1.6227E-08],
    [1.4606E-07, 4.2066E-08, 3.7847E-08, 2.0166E-08],
]
LSTM_SLCD = [
    [],
    [6.03E-01, 6.08E-01, 5.98E-01, 5.45E-01, 5.40E-01, 4.24E-01],
    [6.15E-01, 6.72E-01, 6.94E-01, 5.44E-01, 6.14E-01, 5.39E-01],
    [],
    [2.83E-01, 4.45E-01, 3.60E-01, 3.00E-01, 2.81E-01, 3.19E-01],
    [2.16E-01, 2.77E-01, 3.84E-01, 2.03E-01, 3.26E-01, 1.17E-01],
]

# Validation loss (accuracy for CNN)
CNN_loss = [
    [0.8551, 0.8645, 0.8663, 0.8671],
    [0.8604, 0.8734, 0.8767, 0.8797],
    [0.8615, 0.8720, 0.8764, 0.8770],
    [0.8550, 0.8685, 0.8722, 0.8739],
    [0.8588, 0.8687, 0.8737, 0.8794],
    [0.8602, 0.8686, 0.8736, 0.8766],
]
CNN_SLCD = [
    [],
    [9.39E-04, 1.27E-03, 9.94E-04, 1.05E-03, 1.14E-03, 8.47E-04],
    [2.22E-03, 3.14E-03, 3.63E-03, 1.72E-03, 3.28E-03, 2.06E-03],
    [],
    [1.31E-03, 1.17E-03, 1.13E-03, 5.03E-04, 6.91E-04, 6.08E-04],
    [6.88E-04, 6.37E-04, 8.53E-04, 6.48E-04, 1.02E-03, 6.46E-04],
]

# Placeholder for new models/schedulers (to be filled after experiments)
ResNet_loss = []  # Will be populated after ResNet experiments
ViT_loss = []     # Will be populated after ViT experiments


# Common plot parameters
pparam = dict(
    xlabel='Epochs',
)


def create_bar_plot(ax, data, schedulers, ylabel, ylim=None, yscale='log'):
    x = np.arange(len(epochs))
    n = len(schedulers)
    width = 0.8 / n

    for i, scheduler in enumerate(schedulers):
        if i < len(data):
            ax.bar(x + i*width, data[i], width, label=scheduler)

    ax.set(**pparam, ylabel=ylabel)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(epochs)
    ax.set(yscale=yscale)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.xaxis.set_minor_locator(plt.NullLocator())


def create_grouped_bar_plot(model_name, data_dict, ylabel, yscale='log',
                            ylim=None, filename=None):
    groups = [
        ("Traditional", group_A),
        ("Cyclic/Warmup", group_B),
        ("Hyperbolic", group_C),
    ]

    # Only include groups that have data
    active_groups = []
    for gname, scheds in groups:
        group_data = [data_dict.get(s) for s in scheds if s in data_dict]
        if any(d is not None for d in group_data):
            active_groups.append((gname, scheds))

    if not active_groups:
        return

    ncols = len(active_groups)
    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 2.5), sharey=True)
        if ncols == 1:
            axes = [axes]

        for ax, (gname, scheds) in zip(axes, active_groups):
            avail_scheds = [s for s in scheds if s in data_dict and data_dict[s] is not None]
            avail_data = [data_dict[s] for s in avail_scheds]
            if avail_data:
                create_bar_plot(ax, avail_data, avail_scheds, ylabel if ax == axes[0] else '', ylim, yscale)
            ax.set_title(gname, fontsize=7)

        axes[-1].legend(title='Schedulers', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()


# ── Generate existing plots (backward-compatible) ────────────────────────────

# Build data dicts from existing 6-scheduler data
original_schedulers = ['N', 'P', 'C', 'E', 'H', 'EH']

def make_data_dict(loss_data, schedulers=None):
    if schedulers is None:
        schedulers = original_schedulers
    return {s: loss_data[i] for i, s in enumerate(schedulers) if i < len(loss_data)}


with plt.style.context(["science", "nature"]):
    # Original 2-panel plots (backward compatibility)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2))
    create_bar_plot(ax1, DeepONet_loss, original_schedulers, 'Validation Loss', (1e-5, 2e-2))
    create_bar_plot(ax2, TraONet_loss, original_schedulers, 'Validation Loss', (1e-6, 1e-4))
    ax2.legend(title='Schedulers', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig1.savefig('figs/deeponet_traonet_loss_comparison.png', dpi=600, bbox_inches='tight')

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(6, 2))
    create_bar_plot(ax3, CNN_loss, original_schedulers, 'Accuracy', (0.82, 0.9), 'linear')
    create_bar_plot(ax4, LSTM_loss, original_schedulers, 'Validation Loss', (1e-8, 1e-5))
    ax4.legend(title='Schedulers', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig2.savefig('figs/cnn_lstm_comparison.png', dpi=600, bbox_inches='tight')

    # New grouped plots (will be more useful once new experiments are done)
    create_grouped_bar_plot(
        'CNN', make_data_dict(CNN_loss), 'Accuracy',
        yscale='linear', ylim=(0.82, 0.9),
        filename='figs/cnn_grouped_comparison.png'
    )
    create_grouped_bar_plot(
        'DeepONet', make_data_dict(DeepONet_loss), 'Validation Loss',
        ylim=(1e-5, 2e-2),
        filename='figs/deeponet_grouped_comparison.png'
    )
