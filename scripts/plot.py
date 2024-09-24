#!/usr/bin/env python3
"""
-------------------
MIT License

Copyright (c) 2023  Zeyu Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-------------------
- CIS protection facing multiple attacks
- plot file
-------------------
"""

# import
import numpy as np
import networkx as nx
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt


def plot_G(name, G, solution):
    """
    Plot G
    """
    # node size
    nodesize = np.array([G.nodes[i]['r'] ** 2 for i in G.nodes()])
    nodesize = nodesize / nodesize.sum()
    # figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw_networkx(
        G,
        # distance-based layout
        pos=pos,
        ax=ax, with_labels=True, font_size=14,
        node_size=nodesize * 20000,
        node_color="white",
        edgecolors="black", linewidths=2, width=2,
        edgelist=solution['service'], arrows=True, arrowstyle="-|>"
    )
    nx.draw_networkx_edge_labels(
        G,
        # distance-based layout
        pos=pos,
        edge_labels=solution['service_cost'], font_size=10
    )
    fig.tight_layout()
    fig.savefig(f"figs/{name}.png", dpi=300)
    return


def plot_pi(name, problem, solution):
    """plot pi"""
    # heatmap matrix
    heatmap = np.zeros(shape=(len(problem.G.nodes()), len(problem.S)))
    for i in problem.G.nodes():
        for s in problem.S:
            heatmap[
                len(problem.G.nodes()) - 1 - i, s
            ] = problem.A_lst[solution['policy'][s]][i]
    # figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
    im = ax.imshow(heatmap, cmap="Purples", aspect=2, vmin=-1, vmax=3)
    # Create colorbar
    ax.figure.colorbar(
        im, ax=ax, orientation="horizontal", location="top",
        ticks=[0, 2],
        # values=[0, 1, 2],
        fraction=0.045, label="Policy"
    )
    ax.set_yticks(
        np.arange(heatmap.shape[0]),
        labels=[f'{i}' for i in range(len(problem.G.nodes()) - 1, -1, -1)]
    )
    # label
    ax.set_xlabel("States")
    # white grid
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(heatmap.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(heatmap.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(f"figs/{name}-policy.png", dpi=300)
    return


def plot_V(name, problem, solution):
    """plot V"""
    # sample paths
    path = 20
    heatmap = np.zeros(shape=(len(problem.G.nodes()) + 1, path))
    for k in range(path):
        # initial
        s = [1] * len(problem.G.nodes())
        heatmap[0, k] = solution['value'][problem.S_lst.index(tuple(s))]
        for i in range(1, len(problem.G.nodes()) + 1):
            indices = [j for j in range(len(s)) if s[j] == 1]
            ind = np.random.choice(indices)
            s[ind] = 0
            heatmap[i, k] = solution['value'][problem.S_lst.index(tuple(s))]
    # max v
    v_max = np.max(list(solution['value'].values()))
    # figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    im = ax.imshow(
        heatmap, cmap="Greens", aspect=1, vmin=-500, vmax=v_max
    )
    # Create colorbar
    ax.figure.colorbar(
        im, ax=ax, orientation="horizontal", location="top",
        fraction=0.045, label="State values",
        boundaries=np.linspace(-2000, v_max, 1000),
        ticks=[0, 3500, 7500, 11500, 15000]
    )
    # ticks
    ax.set_xticks(
        np.arange(heatmap.shape[1]),
        labels=[i for i in range(path)]
    )
    ax.set_yticks(
        [i for i in range(len(problem.G.nodes()) + 1)],
        labels=[i for i in range(len(problem.G.nodes()), -1, -1)]
    )
    # label
    ax.set_ylabel("Number of functional CIS")
    ax.set_xlabel("Sample paths")
    # white grid
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(heatmap.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(heatmap.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(f"figs/{name}-value.png", dpi=300)
    return


def plot_G_Knox_urban(name, G, solution):
    """
    Plot G
    """
    # node size
    nodesize = np.array([G.nodes[i]['r'] for i in G.nodes()])
    nodesize = nodesize / nodesize.sum()
    # nodes
    lat, lon = [], []
    for i in G.nodes:
        lat.append(G.nodes[i]['lat'])
        lon.append(G.nodes[i]['lon'])
    # map
    request = cimgt.OSM()
    zoom = 14
    # Bounds: (lon_min, lon_max, lat_min, lat_max):
    extent = [
        -83.99, -83.89305032452431,
        35.9375, 35.999
    ]
    # figure
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(8, 6),
        subplot_kw={'projection': request.crs}
    )
    ax.set_extent(extent)
    ax.add_image(request, zoom)
    # plot edges
    for edge in solution['z'].keys():
        if edge[0] in [0, 1, 2, 3]:
            ax.plot(
                [lon[edge[0]], lon[edge[1]]],
                [lat[edge[0]], lat[edge[1]]],
                transform=ccrs.PlateCarree(),
                linewidth=0.75, color="k",
                linestyle='dashed', zorder=1
            )
    for edge in solution['z'].keys():
        if edge[0] in [0, 1, 2, 3]:
            dx, dy = 0, 0
            dx = 0.0001 * (
                lon[edge[1]] - lon[edge[0]]
            ) / np.abs(
                lon[edge[1]] - lon[edge[0]]
            )
            dy = 0.0001 * (
                lat[edge[1]] - lat[edge[0]]
            ) / np.abs(
                lon[edge[1]] - lon[edge[0]]
            )
            if edge in [(0, 7), (0, 4), (0, 10)]:
                dx = 0.1 * dx
                dy = 0.1 * dy
            ax.arrow(
                (lon[edge[0]] + lon[edge[1]]) / 2,
                (lat[edge[0]] + lat[edge[1]]) / 2,
                dx, dy,
                width=0.00030, linewidth=0, color='black',
                transform=ccrs.PlateCarree()
            )
    # plot nodes
    for i in G.nodes:
        ax.scatter(
            lon[i], lat[i],
            s=600, c='silver', edgecolors='black',
            transform=ccrs.PlateCarree()
        )
    # node
    for i in G.nodes:
        # adjust position
        if i < 10:
            ax.text(
                lon[i] - 0.0008, lat[i] - 0.0008, str(i),
                transform=ccrs.PlateCarree(), fontsize=14
            )
        else:
            ax.text(
                lon[i] - 0.0015, lat[i] - 0.0009, str(i),
                transform=ccrs.PlateCarree(), fontsize=14
            )
        if i < 10:
            # x
            if i == 5:
                ax.text(
                    lon[i] + 0.0025, lat[i] - 0.001,
                    f"${int(solution['x'][i] * (49410 / (52)))}",
                    transform=ccrs.PlateCarree(), fontsize=14, bbox=dict(
                        boxstyle="round", facecolor='white',
                        edgecolor='black', alpha=0.5
                    )
                )
            elif i == 7:
                ax.text(
                    lon[i] + 0.0025, lat[i] + 0.0005,
                    f"${int(solution['x'][i] * (49410 / (52)))}",
                    transform=ccrs.PlateCarree(), fontsize=14, bbox=dict(
                        boxstyle="round", facecolor='white',
                        edgecolor='black', alpha=0.5
                    )
                )
            else:
                ax.text(
                    lon[i] + 0.0025, lat[i] + 0.0025,
                    f"${int(solution['x'][i] * (49410 / (52)))}",
                    transform=ccrs.PlateCarree(), fontsize=14, bbox=dict(
                        boxstyle="round", facecolor='white',
                        edgecolor='black', alpha=0.5
                    )
                )
    fig.tight_layout()
    fig.savefig(f"figs/{name}.png", dpi=300)
    return


def plot_G_Knox_rural(name, G, solution):
    """
    Plot G
    """
    # node size
    nodesize = np.array([G.nodes[i]['r'] for i in G.nodes()])
    nodesize = nodesize / nodesize.sum()
    # nodes
    lat, lon = [], []
    for i in G.nodes:
        if i == 15:
            lat.append(G.nodes[i]['lat'] - 0.005)
            lon.append(G.nodes[i]['lon'] + 0.005)
        else:
            lat.append(G.nodes[i]['lat'])
            lon.append(G.nodes[i]['lon'])
    # map
    request = cimgt.OSM()
    zoom = 14
    # Bounds: (lon_min, lon_max, lat_min, lat_max):
    extent = [
        -83.97047508014354, -84.17647360664639,
        35.882323166191314, 35.980583980764925
    ]
    # figure
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(8, 5),
        subplot_kw={'projection': request.crs}
    )
    ax.set_extent(extent)
    ax.add_image(request, zoom)
    # plot edges
    for edge in solution['z'].keys():
        if edge[0] in [0, 1, 2, 3]:
            ax.plot(
                [lon[edge[0]], lon[edge[1]]],
                [lat[edge[0]], lat[edge[1]]],
                transform=ccrs.PlateCarree(),
                linewidth=0.75, color="k",
                linestyle='dashed', zorder=1
            )
    for edge in solution['z'].keys():
        if edge[0] in [0, 1, 2, 3]:
            dx, dy = 0, 0
            dx = 0.0002 * (
                lon[edge[1]] - lon[edge[0]]
            ) / np.abs(
                lon[edge[1]] - lon[edge[0]]
            )
            dy = 0.0002 * (
                lat[edge[1]] - lat[edge[0]]
            ) / np.abs(
                lon[edge[1]] - lon[edge[0]]
            )
            if edge in [(2, 17), (2, 10), (2, 13)]:
                dx = 0.1 * dx
                dy = 0.1 * dy
            ax.arrow(
                (lon[edge[0]] + lon[edge[1]]) / 2,
                (lat[edge[0]] + lat[edge[1]]) / 2,
                dx, dy,
                width=0.00050, linewidth=0, color='black',
                transform=ccrs.PlateCarree()
            )
    # plot nodes
    for i in G.nodes:
        ax.scatter(
            lon[i], lat[i],
            s=600, c='silver', edgecolors='black',
            transform=ccrs.PlateCarree()
        )
    # node
    for i in G.nodes:
        # adjust position
        if i < 10:
            ax.text(
                lon[i] - 0.0011, lat[i] - 0.0012, str(i),
                transform=ccrs.PlateCarree(), fontsize=14
            )
        else:
            ax.text(
                lon[i] - 0.0025, lat[i] - 0.0012, str(i),
                transform=ccrs.PlateCarree(), fontsize=14
            )
        if i < 10:
            # x
            if i == 9 or i == 7 or i == 6:
                ax.text(
                    lon[i] + 0.004, lat[i] - 0.007,
                    f"${int(solution['x'][i] * (49410 / (52)))}",
                    transform=ccrs.PlateCarree(), fontsize=14, bbox=dict(
                        boxstyle="round", facecolor='white',
                        edgecolor='black', alpha=0.5
                    )
                )
            elif i == 0 or i == 1:
                ax.text(
                    lon[i] - 0.017, lat[i] - 0.008,
                    f"${int(solution['x'][i] * (49410 / (52)))}",
                    transform=ccrs.PlateCarree(), fontsize=14, bbox=dict(
                        boxstyle="round", facecolor='white',
                        edgecolor='black', alpha=0.5
                    )
                )
            else:
                ax.text(
                    lon[i] - 0.023, lat[i] + 0.004,
                    f"${int(solution['x'][i] * (49410 / (52)))}",
                    transform=ccrs.PlateCarree(), fontsize=14, bbox=dict(
                        boxstyle="round", facecolor='white',
                        edgecolor='black', alpha=0.5
                    )
                )
    fig.tight_layout()
    fig.savefig(f"figs/{name}.png", dpi=300)
    return


def plot_pi_Knox(name, problem, solution):
    """plot pi"""
    # heatmap matrix
    heatmap = np.zeros(shape=(len(problem.V), len(problem.S)))
    for i in problem.V:
        for s in problem.S:
            heatmap[
                len(problem.V) - 1 - i, s
            ] = problem.A_lst[solution['pi'][s]][i]
    # figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    sns.heatmap(
        heatmap, cmap="Purples",
        vmin=-3, vmax=7, cbar_kws={
            'orientation': "horizontal", 'location': "top",
            'label': "Policy", 'ticks': [0, 2, 5], 'values': [0, 2, 5],
            'fraction': 0.046
        }
    )
    # ticks
    ax.set_yticks(
        np.arange(heatmap.shape[0]),
        labels=[f'{i}' for i in range(len(problem.V)-1, -1, -1)]
    )
    ax.set_ylabel("CIS Facilities")
    ax.set_xlabel("States")
    fig.tight_layout()
    fig.savefig(f"figs/{name}-policy.png", dpi=300)
    return


def plot_bar_output():
    """
    bar plot, total output
    """
    plt.rc('font', size=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=18)
    plt.rc('figure', titlesize=18)
    # bar plot - total output
    config = ("urban-1", "urban-2", "rural-1", "rural-2")
    output = {
        'conservative policy': (952948, 738145, 972351, 738145),
        'greedy policy': (1090523, 806454, 1140627, 806454),
        'dynamic policy': (1153605, 920888, 1244853, 920888),
    }
    x = np.arange(len(config))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    for attribute, measurement in output.items():
        offset = width * multiplier
        if attribute == 'conservative policy':
            c, h = '#bfcbdb', '//'
        elif attribute == 'greedy policy':
            c, h = '#cccccc', '+'
        else:
            c, h = '#98d1d1', 'x'
        ax.bar(
            x + offset, measurement, width, label=attribute,
            hatch=h, color=c, linewidth=1, edgecolor='black'
        )
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('CIS Facility Output ($)')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, config)
    ax.legend(loc='upper left', ncols=1)
    # ax.legend()
    ax.set_ylim(700000, 1450000)
    fig.tight_layout()
    fig.savefig("figs/output_bar.png", dpi=300)
    return


def plot_bar_loss():
    """
    bar plot, loss
    """
    plt.rc('font', size=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=18)
    plt.rc('figure', titlesize=18)
    # bar plot - loss
    config = ("urban-1", "urban-2", "rural-1", "rural-2")
    loss = {
        'conservative policy': (123191, 200539, 151003, 200539),
        'greedy policy': (61587, 210275, 83951, 210275),
        'dynamic policy': (42775, 125218, 66130, 125218),
    }
    x = np.arange(len(config))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    for attribute, measurement in loss.items():
        offset = width * multiplier
        if attribute == 'conservative policy':
            c, h = '#bfcbdb', '//'
        elif attribute == 'greedy policy':
            c, h = '#cccccc', '+'
        else:
            c, h = '#98d1d1', 'x'
        ax.bar(
            x + offset, measurement, width, label=attribute,
            hatch=h, color=c, linewidth=1, edgecolor='black'
        )
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('CIS Cascading Loss ($)')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, config)
    ax.legend(loc='upper left', ncols=1)
    # ax.legend()
    ax.set_ylim(25000, 325000)
    fig.tight_layout()
    fig.savefig("figs/loss_bar.png", dpi=300)
    return


def plot_bar_success():
    """
    bar plot, successful defenses
    """
    plt.rc('font', size=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=18)
    plt.rc('figure', titlesize=18)
    # bar plot - success
    config = ("urban-1", "urban-2", "rural-1", "rural-2")
    success = {
        'conservative policy': (
            100*5.57/11.12, 100*3.99/11.04, 100*5.52/11.12, 100*3.99/11.04
        ),
        'greedy policy': (
            100*6.74/11.12, 100*3.26/11.31, 100*6.56/11.12, 100*3.26/11.31
        ),
        'dynamic policy': (
            100*7.84/11.12, 100*5.2/11.00, 100*7.79/11.12, 100*5.2/11.00
        ),
    }
    x = np.arange(len(config))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    for attribute, measurement in success.items():
        offset = width * multiplier
        if attribute == 'conservative policy':
            c, h = '#bfcbdb', '//'
        elif attribute == 'greedy policy':
            c, h = '#cccccc', '+'
        else:
            c, h = '#98d1d1', 'x'
        ax.bar(
            x + offset, measurement, width, label=attribute,
            hatch=h, color=c, linewidth=1, edgecolor='black'
        )
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Successful Defenses (%)')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, config)
    ax.legend(loc='upper left', ncols=1)
    # ax.legend()
    ax.set_ylim(20, 100)
    fig.tight_layout()
    fig.savefig("figs/success_bar.png", dpi=300)
    return
