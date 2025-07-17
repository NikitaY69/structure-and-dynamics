import numpy as np
import json
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
plt.style.use('physrev')

import argparse 
from tqdm import tqdm
import os 
import time
import h5py
import glob

# Parser
parser = argparse.ArgumentParser(prog="ABPcorr", description="Compute ABP correlations.")
parser.add_argument(
    "--rootdir", type=str, required=True,
    help="Path to the root directory of the simulation."
)
parser.add_argument(
    "--t0", type=float, default=0,
    help="Wait time (in time units) before starting the analysis."
)
parser.add_argument(
    "--skip", type=float, default=1,
    help="Skip (in time units) between snapshots."
)
parser.add_argument(
    "--sweep", type=float, default=0.1,
    help="Bins width."
)
parser.add_argument(
    "--boxsize", type=float, default=40,
    help="Size of the box where correlations are computed."
)
parser.add_argument(
    "--inf", type=float, default=19.5,
    help="What is considered to be 'infinite' scale-wise."
)
parser.add_argument(
    "--maxB", type=float, default=1,
    help="Maximum value of B(r) in plot."
)
parser.add_argument(
    "--linthresh", type=float, default=0.001,
    help="Linear threshold for norm."
)
parser.add_argument(
    "--linscale", type=float, default=1,
    help="Scale for linear section on norm."
)
parser.add_argument("--remake", action="store_true", help="Remake calculations even if already saved.")
args = parser.parse_args()

t0 = time.time()
# Loading parameters
rootdir = args.rootdir
start = args.t0
remake = args.remake
sweep = args.sweep
box_size = args.boxsize
inf = args.inf
maxB = args.maxB
linthresh = args.linthresh
linscale = args.linscale

json_files = glob.glob(os.path.join(rootdir, "*.json"))
with open(json_files[0], "r") as file:
    params = json.load(file)
N = params["N"]
L = np.sqrt(N/params["density"])
# steps = int(params["tau"]/params["dt"])
# start = params["t0"]


skip = round(args.skip/(params["dt"]*params["skip"]))
# number_of_ss = int((steps-start)/params["skip"])
ss = np.arange(0, params["duration"], params["skip"]*params["dt"])
frames = np.argwhere(ss>=start).flatten()[::skip]
number_of_frames = len(frames)
print(number_of_frames)

# Helper functions
def shift_in_mainbox(a):
    return a - L * np.floor((a + L/2) / (L))

def minimum_image_dist(a, b):
    return L/2 - np.abs(np.abs(a-b)-L/2)

def minimum_image_diff(z1, z2):
    dx = z1.real - z2.real
    dy = z1.imag - z2.imag
    # print(dx.shape)
    # if np.abs(dx) > L/2:
    #     dx -= np.sign(dx) * L
    # if np.abs(dy) > L/2:
    #     dy -= np.sign(dy) * L
    dx -= np.where(np.abs(dx) > L / 2, np.sign(dx) * L, 0)
    dy -= np.where(np.abs(dy) > L / 2, np.sign(dy) * L, 0)

    return dx + 1j * dy

def principal_angle(A):
    return (A + np.pi) % (2 * np.pi) - np.pi

# jPDF parameters
x_edges = np.arange(-box_size/2, box_size/2, sweep)
y_edges = x_edges

if f"corr.npy" not in os.listdir(rootdir) or remake:
    # Loading configurations
    thetas = np.empty(shape=(number_of_frames, N), dtype=float)
    positions = np.empty(shape=(number_of_frames, N), dtype=complex)
    print('Loading configurations ...')
    for i, frame in tqdm(enumerate(frames), total=number_of_frames):
        with h5py.File(f"{rootdir}snapshots.h5", "r") as f:
            data = f["configs"][frame].T
        thetas[i] = data[0]
        pos = shift_in_mainbox(data[1:])
        positions[i] = pos[0] + 1j * pos[1]

    corr = []
    print('Computing correlations ...')
    for i in tqdm(range(N)):
        # Translating origin
        centered_pos = minimum_image_diff(positions, positions[:, [i]])
        # print(centered_pos.real.max(), centered_pos.imag.min())

        # Rotating with respect to self-propulsion direction
        centered_pos = centered_pos * np.exp(-1j * thetas[:, [i]])
        
        # Computing the pair-correlation
        centered_pos = np.delete(centered_pos, i, axis=1)
        # radius = np.abs(centered_pos).flatten()
        # angles = np.angle(centered_pos).flatten()
        # angles = np.angle(centered_pos)-thetas[:, [i]]
        # angles = principal_angle(angles).flatten()
        # corr.append(np.histogram(radius, r_edges)[0])
        X = centered_pos.real.flatten()
        Y = centered_pos.imag.flatten()
        corr.append(np.histogram2d(X, Y, bins=[x_edges, y_edges])[0]/number_of_frames)

    corr = np.sum(np.array(corr), axis=0)
    np.save(f"{rootdir}corr.npy", corr)

else:
    print('Correlations already computed. Loading...')
    corr = np.load(f"{rootdir}corr.npy")

# Renormalization
norm = params["density"]*N*sweep**2# (box_size/sweep)**2 # params["density"]*(box_size/sweep)**2/(N)
m = np.mean(corr[:, np.where(y_edges[:-1] > inf)])
print(m, norm)
corr = corr/norm - 1

# Plot
norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=-maxB, vmax=maxB, base=10)
plt.pcolormesh(x_edges, y_edges, corr.T, shading='auto', cmap='RdBu_r', norm=norm)
plt.colorbar(label=r"$B(\boldsymbol{r})$")

plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.savefig(f"{rootdir}corr.jpg", bbox_inches='tight', dpi=400)
elapsed_time = (time.time() - t0)/60 # minutes
print(f"Execution time: {elapsed_time:.3f} minutes")