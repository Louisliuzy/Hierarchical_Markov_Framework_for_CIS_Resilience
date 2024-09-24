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
- Case study file
-------------------
"""

# import
import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from model import CIS_Defense


def manhattan_dist(A, B):
    """
    Manhattan distance between coordinates
    https://www.usgs.gov/faqs/how-much-distance-does-degree-minute-and-second-cover-your-maps
    """
    return 69 * np.abs(
        A[0] - B[0]
    ) + 54.6 * np.abs(
        A[1] - B[1]
    )


# generate G and Pr
def generate_network(scn, distr, tau):
    """
    generate G and Pr
    """
    # load data
    CIS_data = pd.read_csv(
        "data/Knox_CIS_{}.csv".format(scn), index_col=False
    )
    # construct random network
    G, V, H = nx.DiGraph(), [], {
        1: [], 2: [], 3: [], 4: [], 5: [], 6: []
    }
    # add nodes
    for i in range(CIS_data.shape[0]):
        # type
        H[CIS_data['type'].iloc[i]].append(i)
        if CIS_data['type'].iloc[i] not in [6]:
            V.append(i)
        # add node
        G.add_node(
            i,
            # type
            h=CIS_data['type'].iloc[i],
            # GIS
            lat=CIS_data['lat'].iloc[i],
            lon=CIS_data['lon'].iloc[i],
            # capacity
            C=CIS_data['capacity'].iloc[i],
            # demand
            d={
                1: CIS_data['demnad_power'].iloc[i],
                2: CIS_data['demnad_cellular'].iloc[i],
                3: 0,
                4: CIS_data['demnad_police'].iloc[i],
                5: CIS_data['demnad_fire'].iloc[i],
                6: 0
            },
            # resource cost
            c_r=49410 / (52 * 8),
            # output
            r=CIS_data['output'].iloc[i],
            # threshold
            tau={h: tau for h in [1, 2, 3, 4, 5, 6]},
        )
    # fully connected network, except for manufacturing
    for i in V:
        h_i = G.nodes[i]['h']
        for j in G.nodes():
            if i != j and G.nodes[j]['d'][h_i] > 0:
                # distance
                dist = manhattan_dist(
                    [G.nodes[i]['lat'], G.nodes[i]['lon']],
                    [G.nodes[j]['lat'], G.nodes[j]['lon']]
                )
                if h_i in [1, 2]:
                    c_s = dist * CIS_data['c_s'].iloc[i]
                    c_f = dist * CIS_data['c_f'].iloc[i]
                else:
                    c_s = CIS_data['c_s'].iloc[i]
                    c_f = CIS_data['c_f'].iloc[i]
                G.add_edge(
                    i, j, c_s=c_s,
                    c_f=c_f
                )
    # attack Pr
    Pr = []
    if distr == "weighted":
        for i in V:
            if G.nodes[i]['h'] == 3:
                Pr.append(G.nodes[i]['r'] / 2)
            else:
                Pr.append(G.nodes[i]['r'])
        Pr = Pr / np.sum(Pr)
    else:
        Pr = np.array([1 for i in V])
        Pr = Pr / np.sum(Pr)
    # return
    return G, V, H, Pr


def select_config(config):
    """
    Select configuration
    """
    distr = float("nan")
    scn, tau = float("nan"), float("nan")
    I_a, I_d = float("nan"), float("nan")
    # discount factor, |V|th (final) attack discounted by 25%
    gamma = math.pow(0.75, 1 / (2 * 20))
    if config == 1:
        distr = "weighted"
        scn, tau, runtime = "urban", 0.2, 172800
        I_a, I_d = {1: 0.20, 2: 0.60, 3: 0.20}, [2, 5]
    elif config == 2:
        distr = "even"
        scn, tau, runtime = "urban", 0.2, 172800
        I_a, I_d = {2: 0.20, 6: 0.60, 12: 0.20}, [2, 5]
    elif config == 3:
        distr = "weighted"
        scn, tau, runtime = "rural", 0.2, 172800
        I_a, I_d = {1: 0.20, 2: 0.60, 3: 0.20}, [2, 5]
    elif config == 4:
        distr = "even"
        scn, tau, runtime = "rural", 0.2, 172800
        I_a, I_d = {2: 0.20, 6: 0.60, 12: 0.20}, [2, 5]
    return scn, distr, tau, runtime, I_a, I_d, gamma


def simulate_dynamic(problem, x, z, R, policy):
    """
    Run simulation
    """
    # number of attacks
    attacks = np.random.choice(range(8, 15))
    # history
    state_hist, action_hist, reward_hist = [], [], []
    dependency_loss_hist = []
    defended_attacks_hist = []
    # resources
    resources = {
        i: x[i]
        for i in problem.V
    }
    # initial state
    state = tuple([1] * len(problem.V))
    s = problem.S_lst.index(state)
    state_hist.append(state)
    # total demand
    demand = {
        i: np.sum([problem.G.nodes[i]['d'][j] for j in problem.H.keys()])
        for i in problem.V_D
    }
    # loops
    t = 1
    while True:
        # action
        action = list(problem.A_lst[policy[s]])
        for i in problem.V:
            if action[i] > resources[i]:
                if resources[i] >= 2:
                    action[i] = 2
                else:
                    action[i] = 0
            resources[i] = resources[i] - action[i]
        # index of adjusted action
        a = problem.A_lst.index(tuple(action))
        action_hist.append(action)
        # reward
        reward = R[s]
        reward_hist.append(reward)
        # new epoch
        t += 1
        # new state
        new_state = problem.S_lst[np.random.choice(
            a=list(problem.T[s, a].keys()), size=1, replace=False,
            p=list(problem.T[s, a].values())
        )[0]]
        # defended attacks
        if new_state == state:
            defended_attacks_hist.append(1)
        else:
            defended_attacks_hist.append(0)
        # dependency loss
        V_D_loss = 0
        for j in problem.V_D:
            # percentage input
            loss = 0
            for h in problem.H.keys():
                if any([
                    problem.G.nodes[j]['h'] == h,
                    problem.G.nodes[j]['d'][h]
                 ]) == 0:
                    continue
                if np.sum([
                    z[i, j] for i in problem.H[h] if (i, j) in z.keys()
                ]) < problem.G.nodes[j]['tau'][1] * problem.G.nodes[j]['d'][h]:
                    loss = problem.G.nodes[j]['r']
                    break
            if loss == 0:
                loss = np.max([0, (1 - np.sum([
                    z[i, j] * new_state[i]
                    for i in problem.G.predecessors(j) if (i, j) in z.keys()
                ]) / demand[j])]) * problem.G.nodes[j]['r']
            V_D_loss += loss
        dependency_loss_hist.append(V_D_loss)
        # terminate
        if t > attacks or np.sum(list(new_state)) == 0:
            break
        else:
            # continue
            state = new_state
            s = problem.S_lst.index(state)
            state_hist.append(state)
    # total output
    total_output = np.sum(reward_hist)
    # resource used
    resource_used = np.sum(np.sum(action_hist, axis=1))
    # dependency loss
    dependency_loss = np.sum(dependency_loss_hist)
    # defended attacks
    defended_attacks = np.sum(defended_attacks_hist)
    return [
        total_output, resource_used, np.sum(list(x.values())),
        dependency_loss, defended_attacks, attacks
    ]


def simulate_greedy(problem, x, z, R):
    """
    Run simulation
    """
    # number of attacks
    attacks = np.random.choice(range(8, 15))
    # history
    state_hist, action_hist, reward_hist = [], [], []
    dependency_loss_hist = []
    defended_attacks_hist = []
    # resources
    resources = {
        i: x[i] * 2
        for i in problem.V
    }
    # initial state
    state = tuple([1] * len(problem.V))
    s = problem.S_lst.index(state)
    state_hist.append(state)
    # total demand
    demand = {
        i: np.sum([problem.G.nodes[i]['d'][j] for j in problem.H.keys()])
        for i in problem.V_D
    }
    # loops
    t = 1
    while True:
        # action
        action = [0] * len(problem.V)
        for i in problem.V:
            if resources[i] >= 5:
                action[i] = 5
            else:
                action[i] = 0
            resources[i] = resources[i] - action[i]
        action_hist.append(action)
        for i in problem.V:
            if state[i] == 0:
                action[i] = 0
        # index of adjusted action
        a = problem.A_lst.index(tuple(action))
        # reward
        reward = R[s]
        reward_hist.append(reward)
        # new epoch
        t += 1
        # new state
        new_state = problem.S_lst[np.random.choice(
            a=list(problem.T[s, a].keys()), size=1, replace=False,
            p=list(problem.T[s, a].values())
        )[0]]
        # defended attacks
        if new_state == state:
            defended_attacks_hist.append(1)
        else:
            defended_attacks_hist.append(0)
        # dependency loss
        V_D_loss = 0
        for j in problem.V_D:
            # percentage input
            loss = 0
            for h in problem.H.keys():
                if any([
                    problem.G.nodes[j]['h'] == h,
                    problem.G.nodes[j]['d'][h]
                ]) == 0:
                    continue
                if np.sum([
                    z[i, j] for i in problem.H[h] if (i, j) in z.keys()
                ]) < problem.G.nodes[j]['tau'][1] * problem.G.nodes[j]['d'][h]:
                    loss = problem.G.nodes[j]['r']
                    break
            if loss == 0:
                loss = np.max([0, (1 - np.sum([
                    z[i, j] * new_state[i]
                    for i in problem.G.predecessors(j) if (i, j) in z.keys()
                ]) / demand[j])]) * problem.G.nodes[j]['r']
            V_D_loss += loss
        dependency_loss_hist.append(V_D_loss)
        # terminate
        if t > attacks or np.sum(list(new_state)) == 0:
            break
        else:
            # continue
            state = new_state
            s = problem.S_lst.index(state)
            state_hist.append(state)
    # total output
    total_output = np.sum(reward_hist)
    # resource used
    resource_used = np.sum(np.sum(action_hist, axis=1))
    # dependency loss
    dependency_loss = np.sum(dependency_loss_hist)
    # defended attacks
    defended_attacks = np.sum(defended_attacks_hist)
    return [
        total_output, resource_used, 2 * np.sum(list(x.values())),
        dependency_loss, defended_attacks, attacks
    ]


def simulate_conservative(problem, x, z, R):
    """
    Run simulation
    """
    # number of attacks
    attacks = np.random.choice(range(8, 15))
    # history
    state_hist, action_hist, reward_hist = [], [], []
    dependency_loss_hist = []
    defended_attacks_hist = []
    # resources
    resources = {
        i: x[i] * 2
        for i in problem.V
    }
    # initial state
    state = tuple([1] * len(problem.V))
    s = problem.S_lst.index(state)
    state_hist.append(state)
    # total demand
    demand = {
        i: np.sum([problem.G.nodes[i]['d'][j] for j in problem.H.keys()])
        for i in problem.V_D
    }
    # loops
    t = 1
    while True:
        # action
        action = [0] * len(problem.V)
        for i in problem.V:
            if resources[i] >= 2:
                action[i] = 2
            else:
                action[i] = 0
            resources[i] = resources[i] - action[i]
        action_hist.append(action)
        for i in problem.V:
            if state[i] == 0:
                action[i] = 0
        # index of adjusted action
        a = problem.A_lst.index(tuple(action))
        # reward
        reward = R[s]
        reward_hist.append(reward)
        # new epoch
        t += 1
        # new state
        new_state = problem.S_lst[np.random.choice(
            a=list(problem.T[s, a].keys()), size=1, replace=False,
            p=list(problem.T[s, a].values())
        )[0]]
        # defended attacks
        if new_state == state:
            defended_attacks_hist.append(1)
        else:
            defended_attacks_hist.append(0)
        # dependency loss
        V_D_loss = 0
        for j in problem.V_D:
            # percentage input
            loss = 0
            for h in problem.H.keys():
                if any([
                    problem.G.nodes[j]['h'] == h,
                    problem.G.nodes[j]['d'][h]
                ]) == 0:
                    continue
                if np.sum([
                    z[i, j] for i in problem.H[h] if (i, j) in z.keys()
                ]) < problem.G.nodes[j]['tau'][1] * problem.G.nodes[j]['d'][h]:
                    loss = problem.G.nodes[j]['r']
                    break
            if loss == 0:
                loss = np.max([0, (1 - np.sum([
                    z[i, j] * new_state[i]
                    for i in problem.G.predecessors(j) if (i, j) in z.keys()
                ]) / demand[j])]) * problem.G.nodes[j]['r']
            V_D_loss += loss
        dependency_loss_hist.append(V_D_loss)
        # terminate
        if t > attacks or np.sum(list(new_state)) == 0:
            break
        else:
            # continue
            state = new_state
            s = problem.S_lst.index(state)
            state_hist.append(state)
    # total output
    total_output = np.sum(reward_hist)
    # resource used
    resource_used = np.sum(np.sum(action_hist, axis=1))
    # dependency loss
    dependency_loss = np.sum(dependency_loss_hist)
    # defended attacks
    defended_attacks = np.sum(defended_attacks_hist)
    return [
        total_output, resource_used, 2 * np.sum(list(x.values())),
        dependency_loss, defended_attacks, attacks
    ]


def Knox():
    """
    Benchmarking on instances
    network
    threshold tau
    attack intensity, intensity: probabilty I_a
    defense intensity, list I_d
    """
    config = 0
    scn, distr, tau, runtime, I_a, I_d, gamma = select_config(config)
    # problem name
    name = f"config-{config}"
    # network
    G, V, H, Pr = generate_network(scn, distr, tau)
    # build models
    problem = CIS_Defense(
        name, G, V, H, Pr, I_a, I_d, gamma, runtime
    )
    # NDPI
    solution = problem.NDPI(file_dir="results")
    pickle.dump(solution, open(f"results/{name}-solution.pickle", 'wb'))
    return 0


def main():
    """main"""
    np.random.seed(1)
    # CIS
    Knox()
    return


if __name__ == "__main__":
    main()
