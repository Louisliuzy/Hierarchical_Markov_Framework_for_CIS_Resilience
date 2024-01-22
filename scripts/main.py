#!/usr/bin/env python3
"""
-------------------
MIT License

Copyright (c) 2024  Zeyu Liu

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
- CIS resilience facing sequential attacks
- main file
-------------------
"""

# import
import math
import pickle
import numpy as np
import networkx as nx
import scipy.stats as st
from model import CIS_Defense
import matplotlib.pyplot as plt


# generate G and Pr
def generate_network(V_I, V_D):
    """
    generate G and Pr
    """
    # total edges
    n_edges = np.sum(range(V_I + V_D - 1, 0, -1))
    # density
    density = 0.8
    # construct random network
    G = nx.gnm_random_graph(V_I + V_D, density * n_edges, seed=1)
    # make sure each V_D is at least connected to 1 V_I
    for i in range(V_I, V_I + V_D, 1):
        if all([
            j not in G.neighbors(i)
            for j in range(V_I)
        ]):
            j = np.random.choice(range(V_I))
            G.add_edge(i, j)
    # generate data
    for i in G.nodes.keys():
        # type
        G.nodes[i]['type'] = "I" if i < V_I else "D"
        # resource cost
        G.nodes[i]['c_r'] = st.uniform.rvs(loc=1, scale=4, size=1)[0]
        # output
        G.nodes[i]['r'] = st.uniform.rvs(
            loc=20, scale=60, size=1
        )[0] if i < V_I else st.uniform.rvs(loc=10, scale=90, size=1)[0]
        # dependency proportion
        G.nodes[i]['delta'] = 0.8
    # connection cost
    for edge in G.edges.keys():
        G.edges[edge]['c_s'] = st.uniform.rvs(loc=15, scale=15, size=1)[0]
    # Pr
    Pr = np.array([G.nodes[i]['r'] for i in G.nodes.keys()])
    Pr = Pr / np.sum(Pr)
    # return
    return G, Pr


def plot_G(name, G, solution):
    """
    Plot G
    """
    # node size
    nodesize = np.array([G.nodes[i]['r'] for i in G.nodes()])
    nodesize = nodesize / nodesize.sum()
    # figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    nx.draw_networkx(
        G,
        # distance-based layout
        pos=nx.kamada_kawai_layout(G, weight='c_s'),
        ax=ax, with_labels=True, font_size=14,
        node_size=nodesize * 15000,
        node_color="white",
        edgecolors="black", linewidths=2, width=2,
        edgelist=solution['service'], arrows=True, arrowstyle="-|>"
    )
    nx.draw_networkx_edge_labels(
        G,
        # distance-based layout
        pos=nx.kamada_kawai_layout(G, weight='c_s'),
        edge_labels=solution['service_cost'], font_size=14
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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3.75))
    im = ax.imshow(heatmap, cmap="Purples", aspect=2, vmin=-1.5, vmax=3)
    # Create colorbar
    ax.figure.colorbar(
        im, ax=ax, orientation="horizontal",
        ticks=[0, 1, 2], location="top", values=[0, 1, 2],
        fraction=0.045, label="Policy"
    )
    # ticks
    ax.set_xticks(
        [i for i in range(1, 64, 2)],
        labels=[problem.S_lst[i] for i in range(1, 64, 2)]
    )
    ax.set_yticks(
        np.arange(heatmap.shape[0]),
        labels=[f'CIS {i}' for i in range(len(problem.G.nodes()) - 1, -1, -1)]
    )
    # label
    ax.set_xlabel("States")
    # rotate
    plt.setp(
        ax.get_xticklabels(), rotation=60, ha="right",
        rotation_mode="anchor"
    )
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
        heatmap, cmap="Greens", aspect=1, vmin=-500, vmax=v_max)
    # Create colorbar
    ax.figure.colorbar(
        im, ax=ax, orientation="horizontal", location="top",
        fraction=0.045, label="State values",
        boundaries=np.linspace(0, v_max, 1000),
        ticks=[0, 400, 800, 1200, 1500]
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


def CIS(n_instance=1):
    """
    Benchmarking on instances
    """
    # CIS size
    V_I = 2
    V_D = 4
    # total budget
    B = 100
    # attack intensity, intensity: probabilty
    I_a = {1: 0.50, 2: 0.50}
    # defense intensity
    I_d = [1, 2]
    # discount factor, |V|th (final) attack discounted by 25%
    gamma = math.pow(0.75, 1 / (V_I + V_D))
    # results
    results = {
        'NLP': {}, 'GBD': {}, 'NDPI': {}
    }
    # repeated instances
    for instance in range(n_instance):
        # problem name
        name = f"I={V_I}-D={V_D}-A={len(I_d)}-B={B}-{instance}"
        # network
        G, Pr = generate_network(V_I, V_D)
        # build models
        problem = CIS_Defense(
            name, G, B, Pr, I_a, I_d, gamma
        )
        # NLP
        results_NLP = problem.NLP(file_dir="results")
        results['NLP'][instance] = results_NLP
        # GBD
        results_GBD = problem.GBD(file_dir="results")
        results['GBD'][instance] = results_GBD
        # NDPI
        results_NDPI = problem.NDPI(file_dir="results")
        results['NDPI'][instance] = results_NDPI
    pickle.dump(results, open(
        'results/I={}-D={}-B={}.pickle'.format(V_I, V_D, B), 'wb'
    ))
    return 0


def main():
    """main"""
    np.random.seed(1)
    # CIS
    CIS(n_instance=5)
    return


if __name__ == "__main__":
    main()
