import matplotlib.colors as mcolors
import numpy as np
import colorsys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import PowerNorm

import pandas as pd
import json
import plotly.io as pio
# import plotly.express as px # Not needed for this version
from plotly.subplots import make_subplots # Still needed for title parsing
import re
import os
from pathlib import Path
from matplotlib.colors import Normalize, PowerNorm
import matplotlib.cm as cm

def styleset():
    
    rc = {
        'font.size': 7,
        'font.family': 'Arial',
        'axes.labelsize': 7,    # axis labels size in pt
        'axes.edgecolor': 'k',
        'text.color': 'black',          # General text color
        'axes.labelcolor': 'black',     # X and Y labels color
        'xtick.color': 'black',         # X tick labels (and marks) color
        'ytick.color': 'black',
        'legend.fontsize': 7,
        'legend.title_fontsize': 6, 
        'axes.titlesize': 7,
        'xtick.labelsize': 7,   # tick label font size in pt
        'ytick.labelsize': 7,
        'xtick.major.size': 5,  # tick length in points
        'ytick.major.size': 5,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'pdf.fonttype': 42,
        'figure.autolayout': False,
        "mathtext.fontset": 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', # Bold math text
        'mathtext.sf': 'Arial', # Sans-serif math text
        'mathtext.tt': 'Arial', # Teletype math text
        'mathtext.cal': 'Arial', # Calligraphic math text
        }
    import matplotlib as mpl
    mpl.rcParams.update(rc)
    sns.set(font = 'Arial', rc=rc)
    sns.set_style("ticks", {'axes.edgecolor': 'k',
                            'text.color': 'black',          # Redundancy for safety
                            'xtick.color': 'black',
                            'ytick.color': 'black',
                            'axes.labelcolor': 'black',
                            'axes.linewidth': 1, 
                            'axes.grid': False,
                            'xtick.major.width': 1,
                            'ytick.major.width': 1})
    
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.facecolor"] = "none"
    plt.rcParams["legend.edgecolor"] = "none"
    plt.rcParams["legend.handlelength"] = 1.5
    plt.rcParams["legend.handletextpad"] = 0.4
    plt.rcParams["legend.framealpha"] = 1

    

def lighten(color, amount=0.6):
    color = mcolors.to_rgb(color)
    color = np.array(color)
    white = np.array([1, 1, 1])
    return (1-amount)*color + amount*white

def darken(color, amount=0.2):
    color = mcolors.to_rgb(color)
    color = np.array(color)
    black = np.array([0, 0, 0])
    return (1-amount)*color + amount*black


# discrete palette:
PALETTE = [
    "#6d7fcf", # blue
    "#cc5551", # red
    "#69a558", # green
    "#ca5c8a", # pink
    "#b35ec0", # purple
    "#c2883c", # yellow
]

def show_palette():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PALETTE[i], edgecolor='k', label=f'Class {i+1}') for i in range(len(PALETTE))]
    plt.figure(figsize=(2, 0.5))
    ax = plt.gca()
    plt.legend(handles=legend_elements, loc='center', ncol=len(PALETTE), frameon=False)
    plt.show()


def intensify(color, factor=0.7):
    """Increase color intensity by reducing lightness."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l * factor          # reduce lightness → increases intensity
    return colorsys.hls_to_rgb(h, l, s)

def saturate(color, factor=1.4):
    r, g, b = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(1, s * factor)
    return colorsys.hsv_to_rgb(h, s, v)


intense_green = saturate(darken(PALETTE[2], 0.1), factor=1.3)

HEXPLOT_CMAP = LinearSegmentedColormap.from_list(
    "hexplot_cmap_green",
    ["white", intense_green],
    N=256
)

# other color palettes:
# TODO: could make lambda functions that take in number of colors and return a palette
pal1 = sns.cubehelix_palette(5, rot=-.25, light=.7) #blue
pal2 = sns.cubehelix_palette(5, start=1.90, rot=-0.1, light=.7) #green
pal4 = sns.cubehelix_palette(5, start=1.2, rot=0, light=.9, dark=0.3, hue=0.8) # brown
PINK_CMAP = sns.cubehelix_palette(5, start=0.4, rot=0.2, light=.9, dark=0.2, hue=0.8) # magenta
GRAY_CMAP = sns.cubehelix_palette(5, rot=-.25, light=.7, dark=0.5, hue=0)

# turn this into pd.Interval -- don't use so you can dynamically set bins as you like. 
# valid_bins = [pd.Interval(x, x+0.2, closed='right') for x in np.arange(0, 0.8, 0.2)]
# color_map5 = dict(zip(valid_bins, PINK_CMAP))
# gray_map = dict(zip(valid_bins, GRAY_CMAP))

# CUBEHELIX_CMAP = {
#     "top10_max_tani": color_map5,
#     "seed_bin": gray_map
# }

