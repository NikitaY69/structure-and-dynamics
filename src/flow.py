import numpy as np
import json
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
plt.style.use('physrev')
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator

# from matplotlib.ticker import AutoMinorLocator, LogLocator
# from matplotlib.patches import Rectangle
import argparse 
import h5py
import time
from tqdm import tqdm
import os
import glob

# Parser
parser = argparse.ArgumentParser(prog="ABPf", description="Visualize ABP simulations.")
parser.add_argument(
    "--rootdir", type=str, required=True,
    help="Path to the root directory of the simulation."
)
parser.add_argument(
    "--ntracers", type=int, default=5,
    help="Number of tracers to visualize."
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
    "--fps", type=int, default=10,
    help="Frames per second for the simulation output."
)
parser.add_argument(
    "--selfprop", action='store_true',
    help="Set selfpropulsion as particles' color"
)
parser.add_argument(
    "--msd", action='store_true',
    help="Plot MSD evolution."
)
args = parser.parse_args()
start = args.t0
skip = args.skip

t0 = time.time()
# Loading parameters
rootdir = args.rootdir
json_files = glob.glob(os.path.join(rootdir, "*.json"))
# basename = os.path.basename(os.path.normpath(rootdir))
# f"{rootdir}/{basename}.json"
with open(json_files[0], "r") as file:
    params = json.load(file)

skip = round(skip/(params["dt"]*params["skip"]))
N = params["N"]
L = np.sqrt(N/params["density"])
steps = round(params["duration"]/params["dt"])
# number_of_ss = int(steps/params["skip"])
ss = np.arange(0, params["duration"], params["skip"]*params["dt"])
frames = np.argwhere(ss>=start).flatten()[::skip]
number_of_ss = len(frames)

# Loading cfgs
cfgs = []
print('Loading configurations ...')
for i in tqdm(frames[:-1]):
    with h5py.File(f"{rootdir}snapshots.h5", "r") as f:
        cfgs.append(f["configs"][i].T)

if args.msd:
    obs = np.genfromtxt(f"{rootdir}obs.txt", delimiter='', names=True)
    MSD = obs['MSD']

# Helper functions
def shift_in_mainbox(a):
    return a - L * np.floor((a + L/2) / (L))

# Number of tracer particles
n_tracers = args.ntracers
tracer_indices = np.random.choice(N, n_tracers, replace=False)
tracer_history = {idx: [] for idx in tracer_indices}
colors = np.linspace(0, 1, n_tracers)

# Init fig
if args.msd:
    fig, axes = plt.subplots(1, 2, dpi=300, figsize=(12, 6))
else:
    fig, axes = plt.subplots(1, 1, dpi=300, figsize=(6, 6))
ax_main = axes[0] if args.msd else axes
ax_msd = axes[1] if args.msd else None
plt.tight_layout()
ax_main.set_facecolor('black')

# particles = ax_main.scatter([], [], s=np.sqrt(N), color='linen', alpha=0.5, linewidths=0.5)

if not args.selfprop:
    particles = [Circle((0, 0), 0.5, facecolor='linen', linewidth=0.5, alpha=0.5) \
                for _ in range(N)]

    for i, idx in enumerate(tracer_indices):
        particles[idx].set_radius(1.0)
        particles[idx].set_facecolor(cm.rainbow(colors[i]))
        particles[idx].set_alpha(1)
    particles_c = PatchCollection(particles, match_original=True)
    tracer_lines = LineCollection([], linewidths=1, alpha=0.75, cmap='rainbow')
else:
    particles = [Circle((0, 0), 0.5, linewidth=0.5, alpha=0.5) \
                for _ in range(N)]

    for i, idx in enumerate(tracer_indices):
        particles[idx].set_radius(1.0)
        particles[idx].set_facecolor('black')
        particles[idx].set_alpha(1)
    particles_c = PatchCollection(particles, cmap='jet', match_original=True)
    particles_c.set_clim(vmin=0, vmax=2*np.pi)
    tracer_lines = LineCollection([], linewidths=1, alpha=0.75, color='k')
    divider = make_axes_locatable(ax_main)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(particles_c, cax=cbar_ax, label=r'$\theta_i$', ticks=[0, np.pi, 2*np.pi])
    cbar.ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    cbar.ax.yaxis.set_minor_locator(MultipleLocator(base=np.pi/2))

ax_main.add_collection(particles_c)
ax_main.add_collection(tracer_lines)

# Init MSD
if args.msd:
    ax_msd.set_xlabel(r"$t$")
    ax_msd.set_xscale('log')
    ax_msd.set_yscale('log')
    ax_msd.set_title("MSD")
    ax_msd.set_autoscale_on(True)

    msd_line, = ax_msd.plot([], [], color='k', label=r'$\langle \Delta r^2 \rangle$')
    msd_lines = [ax_msd.plot([], [], color=plt.cm.rainbow(c), marker='none', lw=1)[0] for c in colors]
    ax_msd.legend(loc='upper left')

    msd_data = {idx: [] for idx in tracer_indices}
    time_data = []
    initial_positions = {}

def init():
    # Limits
    ax_main.set_xlim([-L/2, L/2])
    ax_main.set_ylim([-L/2, L/2])    
    # Remove the border (spines)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    # Remove ticks
    # ax.add_patch(Rectangle((-L/2, -L/2), width=L, height=L, color='k', fill=False, linewidth=1))
    ax_main.xaxis.set_ticks([])
    ax_main.yaxis.set_ticks([])
    ax_main.set_xticks([-L/2, L/2])
    if args.msd:
        return particles_c, tracer_lines, *msd_lines, msd_line, 
    else:
        return particles_c, tracer_lines,

segments = []
segment_colors = []
def update(t):
    # s = int(t/params["dt"])
    if args.msd:
        time_data.append(ss[frames[t]])
    ax_main.set_title(r"$t=$"f"{ss[frames[t]]:.2f}")    

    # Positions
    cfg = cfgs[t]
    pos = np.vstack((shift_in_mainbox(cfg[1]), shift_in_mainbox(cfg[2]))).T
    for i, p in enumerate(particles):
        p.set_center((pos[i, 0], pos[i, 1]))

    particles_c.set_paths(particles)
    if args.selfprop:
        particles_c.set_array(cfg[0] % (2*np.pi))
    # scat.set_offsets(pos)

    # Tracers
    for i, idx in enumerate(tracer_indices):
        xfull, yfull = cfg[1][idx], cfg[2][idx]
        x, y = pos[idx]
        tracer_history[idx].append((x, y))
        
        if len(tracer_history[idx]) > 1:
            segment = np.array(tracer_history[idx][-2:])
            segment2 = segment.copy()
            dx, dy = segment[1] - segment[0]
            
            # Dealing with PBCs
            if np.abs(dx) > L / 2:
                if dx > 0:
                    segment[1, 0] -= L
                    segment2[0, 0] += L
                else:
                    segment[1, 0] += L
                    segment2[0, 0] -= L
            
            if np.abs(dy) > L / 2:
                if dy > 0:
                    segment[1, 1] -= L
                    segment2[0, 1] += L
                else:
                    segment[1, 1] += L
                    segment2[0, 1] -= L
            
            segments.append(segment)
            segments.append(segment2)
            segment_colors.append(colors[i])
            segment_colors.append(colors[i])

        # MSD
        if args.msd:
            if idx not in initial_positions:
                initial_positions[idx] = np.array([xfull, yfull])

            dx = xfull - initial_positions[idx][0]
            dy = yfull - initial_positions[idx][1]

            msd = dx**2 + dy**2
            msd_data[idx].append(msd)

            msd_lines[i].set_data(time_data, msd_data[idx])
            msd_line.set_data(time_data, MSD[:len(time_data)])

    tracer_lines.set_segments(segments)
    tracer_lines.set_array(np.array(segment_colors)) 

    # Autoscale
    if args.msd:
        ax_msd.relim()
        ax_msd.autoscale_view(True, True, True) 
        return particles_c, tracer_lines, *msd_lines, msd_line,
    else:
        return particles_c, tracer_lines,

fig.tight_layout()
print('Making movie...')
anim = animation.FuncAnimation(fig, update, init_func=init, frames=tqdm(range(number_of_ss-1)), blit=True)
writermp4 = animation.FFMpegWriter(fps=args.fps)
anim.save(f"{rootdir}flow.mp4", writer=writermp4)
elapsed_time = (time.time() - t0)/60 # minutes
print(f"Execution time: {elapsed_time:.3f} minutes")
