import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import seaborn as sns

n_operators = np.asarray([18, 38, 74, 126, 194])

total_scoped_time_mean = np.asarray([3.973349e+02, 4.005533e+02, 4.011255e+02, 3.924873e+02, 4.056264e+02])/60
total_scoped_time_std = np.asarray([20.122303, 15.387792, 13.283030, 5.894847, 13.731410])/60
# total_scoped_time_cv = [ 0.050643, 0.038416, 0.033114, 0.015019, 0.033852]

total_unscoped_time_mean = np.asarray([7.028990e+02, 1.440439e+03, 1.980956e+03, 2.685725e+03, 3.509843e+03])/60
total_unscoped_time_std = np.asarray([13.077346, 63.509820, 25.234270, 114.599999, 121.143835])/60
# total_unscoped_time_cv = [0.018605, 0.044091, 0.012738, 0.042670, 0.034515]

y_vals = reversed([total_scoped_time_mean, total_unscoped_time_mean])
err_vals = reversed([total_scoped_time_std, total_unscoped_time_std])
c_vals = reversed(['b', 'r'])
label_vals = reversed(['ENHSP + Task Scoping', 'ENHSP'])

# example data
x = n_operators
fig, ax = plt.subplots()
for y, error, c, l in zip(y_vals, err_vals, c_vals, label_vals):
    # example error bar values that vary with x-position
    # error bar values w/ different -/+ errors
    lower_error = y - error
    upper_error = y + error
    asymmetric_error = [lower_error, upper_error]

    ax.errorbar(x, y, yerr=error, fmt='-o', c=c, label=l)
    ax.fill_between(x, lower_error, upper_error, color=c, alpha=0.2)
    ax.set_title('Planning Time vs. Operators')# (Log-scale)')
    # ax.set_xscale('log')
    # ax.set_xlim([10,210])
    subs = [1.0, 2.0, 3.0, 6.0]
    # ax.tick_params(direction='out', length=6, width=2, colors='r',
               # grid_color='r', grid_alpha=0.5)
    # ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subs)) #set the ticks position
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    # ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_ticks(x)

    ax.set_xlabel('Number of Operators')# (Log-spacing)')
    ax.set_ylabel('Planning Time (minutes)')
plt.minorticks_off()
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('planning-time-vs-operators.png', facecolor='w')
plt.show()
