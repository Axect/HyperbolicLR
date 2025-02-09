import matplotlib.pyplot as plt
import scienceplots
import numpy as np

# Prepare Data to Plot
epochs = [50, 100, 150, 200]
models = ['TraONet', 'DeepONet', 'LSTM', 'CNN']
schedulers = ['N', 'P', 'C', 'E', 'H', 'EH']


# Validation loss
TraONet_loss    = [
    [5.0977E-05,4.1721E-05,2.2007E-05,1.0926E-05],
    [2.8082E-05,8.9639E-06,5.2343E-06,4.0906E-06],
    [3.4426E-05,9.2909E-06,5.2014E-06,4.1982E-06],
    [3.5151E-05,1.6451E-05,1.2681E-05,1.1541E-05],
    [3.1015E-05,8.9335E-06,2.8333E-06,1.3565E-06],
    [2.9069E-05,9.9736E-06,4.9011E-06,2.8939E-06],
]
# Diff: mean, std
TraONet_diff    = [
    [38.59, 17.76],
    [43.85, 23.20],
    [45.44, 26.89],
    [28.37, 22.60],
    [63.87, 10.28],
    [52.50, 12.45],
]
# y = exp(A) * x**B
TraONet_reg_coeff = [
    [-5.5519, -1.0576],
    [-5.0035, -1.4159],
    [-4.2787, -1.5561],
    [-7.1059, -0.8229],
    [-1.4219, -2.2630],
    [-3.9291, -1.6585],
]
# Regression stat (R2, Adj-R2, p-value)
TraONet_reg_stat = [
    [0.8344, 0.7517, 0.0865],
    [0.9877, 0.9815, 0.0062],
    [0.9789, 0.9683, 0.0106],
    [0.9601, 0.9402, 0.0202],
    [0.9878, 0.9817, 0.0061],
    [0.9983, 0.9975, 0.0008],
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

# Diff: mean, std
DeepONet_diff = [
    [16.26, 56.74],
    [-401.69, 779.00],
    [-970.47, 1034.60],
    [19.56, 16.23],
    [-2304.03, 4072.73],
    [28.83, 17.06],
]

# y = exp(A) * x**B
DeepONet_reg_coeff = [
    [-5.8570, -0.8175],
    [-13.4221, 1.0624],
    [-25.1690, 3.7606],
    [-7.7004, -0.5272],
    [-18.2947, 2.2412],
    [-6.3750, -0.8172],
]

# Regression stat (R2, Adj-R2, p-value)
DeepONet_reg_stat = [
    [0.5618, 0.3427, 0.2505],
    [0.3084, -0.0374, 0.4447],
    [0.8757, 0.8135, 0.0642],
    [0.9682, 0.9522, 0.0161],
    [0.4806, 0.2209, 0.3068],
    [0.8949, 0.8423, 0.0540],
]

# SLCD (Smoothed Learning Curve Difference)
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

# Diff: mean, std
LSTM_diff = [
    [-37.57, 95.42],
    [-14.20, 146.11],
    [30.22, 47.61],
    [17.84, 21.80],
    [65.51, 22.96],
    [42.65, 30.79],
]

# y = exp(A) * x**B
LSTM_reg_coeff = [
    [-14.4329, 0.4879],
    [-12.9833, -0.8070],
    [-12.4665, -0.9662],
    [-14.1844, -0.4982],
    [-2.8726, -2.8715],
    [-10.5617, -1.3431],
]

# Regression stat (R2, Adj-R2, p-value)
LSTM_reg_stat = [
    [0.4688, 0.2032, 0.3153],
    [0.2782, -0.0827, 0.4726],
    [0.8342, 0.7513, 0.0867],
    [0.8882, 0.8323, 0.0576],
    [0.9799, 0.9698, 0.0101],
    [0.9493, 0.9239, 0.0257],
]

# SLCD (Smoothed Learning Curve Difference)
LSTM_SLCD = [
    [],
    [6.03E-01, 6.08E-01, 5.98E-01, 5.45E-01, 5.40E-01, 4.24E-01],
    [6.15E-01, 6.72E-01, 6.94E-01, 5.44E-01, 6.14E-01, 5.39E-01],
    [],
    [2.83E-01, 4.45E-01, 3.60E-01, 3.00E-01, 2.81E-01, 3.19E-01],
    [2.16E-01, 2.77E-01, 3.84E-01, 2.03E-01, 3.26E-01, 1.17E-01],
]

# Validation loss (assuming these are accuracy values instead of loss)
CNN_loss = [
    [0.8551, 0.8645, 0.8663, 0.8671],
    [0.8604, 0.8734, 0.8767, 0.8797],
    [0.8615, 0.8720, 0.8764, 0.8770],
    [0.8550, 0.8685, 0.8722, 0.8739],
    [0.8588, 0.8687, 0.8737, 0.8794],
    [0.8602, 0.8686, 0.8736, 0.8766],
]

# Diff: mean, std
CNN_diff = [
    [0.0040, 0.0047],
    [0.0064, 0.0057],
    [0.0052, 0.0050],
    [0.0063, 0.0063],
    [0.0069, 0.0026],
    [0.0055, 0.0028],
]

# y = exp(A) * x**B
CNN_reg_coeff = [
    [-0.1950, 0.0102],
    [-0.2116, 0.0160],
    [-0.2004, 0.0134],
    [-0.2178, 0.0160],
    [-0.2177, 0.0167],
    [-0.2042, 0.0137],
]

# Regression stat (R2, Adj-R2, p-value)
CNN_reg_stat = [
    [0.9139, 0.8709, 0.0440],
    [0.9639, 0.9459, 0.0182],
    [0.9543, 0.9314, 0.0231],
    [0.9467, 0.9200, 0.0270],
    [0.9949, 0.9923, 0.0026],
    [0.9990, 0.9985, 0.0005],
]

# SLCD (Smoothed Learning Curve Difference)
CNN_SLCD = [
    [],
    [9.39E-04, 1.27E-03, 9.94E-04, 1.05E-03, 1.14E-03, 8.47E-04],
    [2.22E-03, 3.14E-03, 3.63E-03, 1.72E-03, 3.28E-03, 2.06E-03],
    [],
    [1.31E-03, 1.17E-03, 1.13E-03, 5.03E-04, 6.91E-04, 6.08E-04],
    [6.88E-04, 6.37E-04, 8.53E-04, 6.48E-04, 1.02E-03, 6.46E-04],
]


# Common plot parameters
pparam = dict(
    xlabel='Epochs',
)


# Function to create bar plot
def create_bar_plot(ax, model, data, ylabel, ylim=None, yscale='log'):
    x = np.arange(len(epochs))
    width = 0.14  # Width of each bar
    
    for i, scheduler in enumerate(schedulers):
        ax.bar(x + i*width, data[i], width, label=scheduler)
    
    #ax.set_title(f'{model}')
    ax.set(**pparam, ylabel=ylabel)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(epochs)
    ax.set(yscale=yscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Remove minor ticks
    ax.xaxis.set_minor_locator(plt.NullLocator())


# Set up the plots
with plt.style.context(["science", "nature"]):
    # Plot for DeepONet and TraONet
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2))

    create_bar_plot(ax1, 'DeepONet', DeepONet_loss, 'Validation Loss', (1e-5, 2e-2))
    create_bar_plot(ax2, 'TraONet', TraONet_loss, 'Validation Loss', (1e-6, 1e-4))
    
    ax2.legend(title='Schedulers', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    fig1.savefig('figs/deeponet_traonet_loss_comparison.png', dpi=600, bbox_inches='tight')

    # Plot for CNN and LSTM
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(6, 2))

    create_bar_plot(ax3, 'SimpleCNN', CNN_loss, 'Accuracy', (0.82, 0.9), 'linear')
    create_bar_plot(ax4, 'LSTM Seq2Seq', LSTM_loss, 'Validation Loss', (1e-8, 1e-5))
    
    ax4.legend(title='Schedulers', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    fig2.savefig('figs/cnn_lstm_comparison.png', dpi=600, bbox_inches='tight')
