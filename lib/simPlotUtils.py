description = """a module that holds helper functions for plotting and visualization for simulations and subsequent skymap analysis"""
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    "axes.grid": True,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "figure.subplot.bottom": 0.17,
    "grid.color": 'gray',
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

from lalinference import plot as lalinf_plot ### load in here to avoid conficts with dependencies
from lalinference import cmap

import numpy as np
import healpy as hp

import os
import json

import simUtils as utils

#-------------------------------------------------

def mollweideScaffold( post, post_masked, mask=None, projection="astro mollweide", contours=False, color_map='OrRd', contour_color='grey', LatLon=None):
    """
    a standardized plotting routine for the mollweide projections needed in simPlot.py
    each mask supplied via *masks is overlayed on the masked plot

    LatLon will be marked with a red dot iff projection=="mollweide" (assumed EarthFixed coords)
    """
    ### set up figure and axis objects
    figwidth = 10
    figheight = 10

    fig = plt.figure(figsize=(figwidth,figheight))
    ax1 = plt.subplot(2,1,1, projection=projection)
    ax2 = plt.subplot(2,1,2, projection=projection)

    ### establish color map
    cmap = plt.get_cmap(color_map)

    ### establish range for color map
    vmin = min(np.min(post), np.min(post_masked))
    vmax = max(np.max(post), np.max(post_masked))

    ### plot heatmaps
    plt.sca( ax1 )
    lalinf_plot.healpix_heatmap( post, cmap=cmap, vmin=vmin, vmax=vmax, nest=False )
    
    plt.sca( ax2 )
    lalinf_plot.healpix_heatmap( post_masked, cmap=cmap, vmin=vmin, vmax=vmax, nest=False )

    if mask!=None: ### plot the mask as a shaded region
        mask = np.ma.masked_where(mask==1, mask)
        lalinf_plot.healpix_heatmap( mask, cmap=plt.get_cmap('Greys_r'), vmin=0, vmax=1, nest=False, alpha=0.1 )

    if projection=="mollweide": ### add continents
        geojson_filename = os.path.join(os.path.dirname(lalinf_plot.__file__),
            'ne_simplified_coastline.json')
        with open(geojson_filename, 'r') as geojson_file:
            geojson = json.load(geojson_file)
        for shape in geojson['geometries']:
            verts = np.deg2rad(shape['coordinates'])
            color='k'
            ax1.plot(verts[:, 0], verts[:, 1], color=color, linewidth=0.5)
            ax2.plot(verts[:, 0], verts[:, 1], color=color, linewidth=0.5)

    ### plot contours
    if contours:
        if np.sum(post):
            cpost = np.empty(post.shape)
            indecies = np.argsort(post)[::-1]
            cpost[indecies] = np.cumsum(post[indecies])

            plt.sca( ax1 )
            lalinf_plot.healpix_contour( cpost, alpha=0.25, levels=[0.1, 0.5, 0.9], colors=contour_color )
        if np.sum(post_masked):
            cpost = np.empty(post_masked.shape)
            indecies = np.argsort(post_masked)[::-1]
            cpost[indecies] = np.cumsum(post_masked[indecies])

            plt.sca( ax2 )
            lalinf_plot.healpix_contour( cpost, alpha=0.25, levels=[0.1, 0.5, 0.9], colors=contour_color )

    ### set titles
    ax1.set_title('posterior')
    ax2.set_title('masked posterior')

    ### decoration
    for ax in [ax1, ax2]:
        if LatLon!=None:
            lat, lon = LatLon
            ax.plot( lon, lat, marker='o', markeredgecolor='r', markerfacecolor='r', markersize=3, linestyle='none')
        ax.patch.set_alpha(0.)
        ax.set_alpha(0.)

    ### return fig and ax handles
    return fig, [ax1, ax2]
