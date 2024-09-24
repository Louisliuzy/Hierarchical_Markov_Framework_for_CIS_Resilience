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
- CIS protection facing multiple attacks
- main file
-------------------
"""

# import
import math
import numpy as np
import networkx as nx
import scipy.stats as st
from model import CIS_Defense


# generate G and Pr
def generate_network(V, H, tau):
    """
    generate G and Pr
    """
    # fully connected network
    G = nx.DiGraph()
    for i in range(V):
        G.add_node(i)
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                G.add_edge(i, j)
    # generate data
    for i in G.nodes():
        # type
        for h in H.keys():
            if i < H[h]:
                G.nodes[i]['h'] = h
                break
        # demand
        if G.nodes[i]['h'] == 1:
            G.nodes[i]['d'] = {h: 0 for h in H.keys()}
        else:
            G.nodes[i]['d'] = {
                h: st.uniform.rvs(
                    loc=5, scale=10, size=1
                )[0] if h != G.nodes[i]['h'] else 0
                for h in H.keys()
            }
        # supply
        G.nodes[i]['C'] = st.uniform.rvs(loc=50, scale=50, size=1)[0]
        # resource cost
        G.nodes[i]['c_r'] = st.uniform.rvs(loc=1, scale=5, size=1)[0]
        # output
        G.nodes[i]['r'] = st.uniform.rvs(loc=50, scale=100, size=1)[0]
        # threshold
        G.nodes[i]['tau'] = {h: tau for h in H.keys()}
        # dependency proportion
        # G.nodes[i]['delta'] = 0.8
    # connection & service costs
    for edge in G.edges():
        G.edges[edge]['c_f'] = st.uniform.rvs(loc=5, scale=20, size=1)[0]
        G.edges[edge]['c_s'] = G.edges[edge]['c_f'] / 10
    # attack Pr
    Pr = np.array([G.nodes[i]['r'] for i in G.nodes()])
    Pr = Pr / np.sum(Pr)
    # return
    return G, Pr


def select_config(config):
    """
    Select configuration
    """
    V, tau = float("nan"), float("nan")
    I_a, I_d = float("nan"), float("nan")
    if config == 1:
        V, tau, runtime = 6, 0.0, 1800
        I_a, I_d = {1: 0.30, 2: 0.40, 3: 0.30}, [2]
    elif config == 2:
        V, tau, runtime = 6, 0.2, 1800
        I_a, I_d = {1: 0.30, 2: 0.40, 3: 0.30}, [2]
    elif config == 3:
        V, tau, runtime = 6, 0.0, 1800
        I_a, I_d = {1: 0.30, 2: 0.40, 3: 0.30}, [2, 4]
    elif config == 4:
        V, tau, runtime = 9, 0.0, 3600
        I_a, I_d = {1: 0.30, 2: 0.40, 3: 0.30}, [2]
    elif config == 5:
        V, tau, runtime = 9, 0.2, 3600
        I_a, I_d = {1: 0.30, 2: 0.40, 3: 0.30}, [2]
    elif config == 6:
        V, tau, runtime = 9, 0.0, 3600
        I_a, I_d = {1: 0.30, 2: 0.40, 3: 0.30}, [2, 4]
    elif config == 7:
        V, tau, runtime = 12, 0.0, 7200
        I_a, I_d = {1: 0.30, 2: 0.40, 3: 0.30}, [2]
    elif config == 8:
        V, tau, runtime = 12, 0.0, 7200
        I_a, I_d = {1: 0.30, 2: 0.40, 3: 0.30}, [2, 4]
    return V, tau, runtime, I_a, I_d


def CIS(n_instance=1):
    """
    Benchmarking on instances
    CIS size V
    threshold tau
    attack intensity, intensity: probabilty I_a
    defense intensity, list I_d
    """
    config = 1
    V, tau, runtime, I_a, I_d = select_config(config)
    # types: number
    H = {h: (h / 3) * V for h in range(1, 4)}
    # discount factor, |V|th (final) attack discounted by 25%
    gamma = math.pow(0.75, 1 / (2 * V))
    # results
    results = {'NLP': {}, 'GBD': {}, 'NDPI': {}}
    # repeated instances
    for instance in range(n_instance):
        # problem name
        name = f"config-{config}-{instance}"
        # generate network
        G, Pr = generate_network(V, H, tau)
        # build models
        problem = CIS_Defense(
            name, H, G, Pr, I_a, I_d, gamma, runtime
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
    return 0


def main():
    """main"""
    np.random.seed(1)
    # CIS
    CIS(n_instance=1)
    return


if __name__ == "__main__":
    main()
