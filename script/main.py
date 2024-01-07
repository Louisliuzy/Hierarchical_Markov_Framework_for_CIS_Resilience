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
