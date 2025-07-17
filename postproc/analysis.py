import numpy as np
import json
from matplotlib import cm
import matplotlib.pyplot as plt
plt.style.use('physrev')
from matplotlib.ticker import AutoMinorLocator, LogLocator
import argparse 
from tqdm import tqdm
import os
import glob

# Parser
parser = argparse.ArgumentParser(
    prog="ABPobs", 
    description=("Plot ABP observables from a finished simulation.\n"
                 "\n"
                 "Automatically loads the observables from the root directory and plots " \
                 "them in the same directory."
    )
)
parser.add_argument(
    "--rootdir", type=str, required=True,
    help="Path to the root directory of the simulation."
)
parser.add_argument(
    "--observe", type=str, required=True,
    help="Observable to be plotted."
)

def main():
    args = parser.parse_args()

    # Loading parameters
    rootdir = args.rootdir
    observe = args.observe
    json_files = glob.glob(os.path.join(rootdir, "*.json"))
    with open(json_files[0], "r") as file:
        params = json.load(file)

    N = params["N"]
    L = np.sqrt(N/params["density"])
    # steps = int(params["tau"]/params["dt"])
    start = 0
    # number_of_ss = int((steps-start)/params["skip"])
    ss = np.arange(start, params["duration"]-params["t0"], params["skip"]*params["dt"])
    if observe == "MSD":
        t0 = 0
    elif observe == "angularAC" or observe == "2ptMemory":
        t0 = -1

    idx = np.argwhere(ss>t0)
    ss = ss[idx]-t0
    number_of_ss = len(ss)
    print(number_of_ss)

    # configs = np.empty(shape=(number_of_ss, 3, N))
    # for step in range(number_of_ss):
    #     print(step)
    #     configs[step] = np.genfromtxt(f"{rootdir}/configs/cfg_{step}.xy").T

    # Loading the observables on log-spaced grids
    data = np.genfromtxt(f"{rootdir}obs.txt", delimiter='', names=True)
    print(number_of_ss, params['duration'], data.shape)
    logspaces = np.unique(np.logspace(0, np.log10(number_of_ss), 100, endpoint=False, dtype=int))
    if observe == "MSD":
        y = data[observe][idx]# [logspaces]
        MSD_theta = data['angularMSD'][idx]# [logspaces]
    elif observe == "angularAC" or observe == "2ptMemory":
        y = data[observe][idx]

    ts = ss # np.linspace(0, params["tau"], number_of_ss)[logspaces]
    ylabels = {'MSD':r'$\left\langle\left(X(t)-X(0)\right)^2\right\rangle$',
            'angularAC': r'$\left\langle\mathbf{e}_{\theta}(0)\cdot\mathbf{e}_{\theta}(t)\right\rangle$',
            '2ptMemory': r'$d(0,t)$'}

    plotlabels = {'MSD':r'$\mathbf{r}$',
                'angularAC': None,
                '2ptMemory': None}
    # Plot
    fig, ax = plt.subplots()
    plt.title(r'$\mathrm{Pe}=$' + f'{3*params["U"]/params["D0_r"]:.1f}' + 
            r',$D_0/D_s=$' + f'{params["D0"]/params["Ds"]:.1f}' + 
            r',$\tau=$' + f'{params["tau"]:.1f}')

    if observe == "MSD":
        plt.loglog(ts, y, ms=6, color='blue', label=plotlabels[observe])
        plt.loglog(ts, MSD_theta, ms=6, color='red', 
                label=r'$\theta$')
    elif observe == "angularAC":
        plt.semilogx(ts, y, ms=6, marker='none', color='blue', label=plotlabels[observe])
    elif observe == "2ptMemory":
        plt.plot(ts, y, ms=6, marker='none', color='blue', label=plotlabels[observe])

    plt.xlabel(r'$t$')
    plt.ylabel(ylabels[observe])
    plt.legend(frameon=False)

    if observe == "MSD" or observe == "angularAC":
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    elif observe == "2ptMemory":
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if observe == "MSD":
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    elif observe == "angularAC" or observe == "2ptMemory":
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    plt.savefig(f"{rootdir}{observe}.jpg", bbox_inches='tight', dpi=400)

if __name__ == "__main__":
    main()