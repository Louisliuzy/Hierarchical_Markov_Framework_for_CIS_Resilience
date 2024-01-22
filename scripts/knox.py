# import
import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from knox_model import Knox_Defense


def haversine_dist(A, B):
    """haversine_dist"""
    # Earth radius in km
    R = 6371
    dLat, dLon = np.radians(B[0] - A[0]), np.radians(B[1] - A[1])
    lat1, lat2 = np.radians(A[0]), np.radians(B[0])
    a = np.sin(dLat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon/2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


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
def generate_network(config):
    """
    generate G and Pr
    """
    # load data
    CIS_data = pd.read_csv(
        "data/Knox_CIS_{}.csv".format(config), index_col=False
    )
    # construct random network
    G = nx.Graph()
    # add nodes
    for i in range(CIS_data.shape[0]):
        G.add_node(
            i,
            # type
            type=CIS_data['type'].iloc[i],
            # GIS
            lat=CIS_data['lat'].iloc[i],
            lon=CIS_data['lon'].iloc[i],
            # resource cost
            c_r=49.410 / (52),
            # output
            r=24 * CIS_data['output'].iloc[i],
            # dependency proportion
            delta=0.8
        )
    # add edges
    line_cost = 31.44  # EIA
    for i in G.nodes.keys():
        for j in G.nodes.keys():
            if i != j and (i, j) not in G.edges():
                # distance
                dist = haversine_dist(
                    [G.nodes[i]['lat'], G.nodes[i]['lon']],
                    [G.nodes[j]['lat'], G.nodes[j]['lon']]
                )
                # add edge
                G.add_edge(i, j, c_s=dist * line_cost)
    # Pr
    Pr = np.array([G.nodes[i]['r'] for i in G.nodes.keys()])
    Pr = Pr / np.sum(Pr)
    # return
    return G, Pr


def main():
    # CIS size
    config = "rural"
    size = 12
    # total budget
    B = 1500
    profile = 2
    # attack intensity, intensity: probabilty
    I_a = {2: 0.20, 6: 0.60, 12: 0.20}
    # defense intensity
    I_d = [2, 5]
    # discount factor, |V|th (final) attack discounted by 25%
    gamma = math.pow(0.95, 1 / size)
    # problem name
    name = f"{config}-{profile}"
    # network
    G, Pr = generate_network(config)
    # build models
    problem = Knox_Defense(
        name, G, B, Pr, I_a, I_d, gamma
    )
    # NDPI
    # results_NDPI = problem.NDPI(file_dir="results")
    results_NDPI = problem.NDPI(file_dir="results")
    pickle.dump(results_NDPI, open(f'results/{name}.pickle', 'wb'))
    # simulation
    results_NDPI = pickle.load(open(f'results/{name}.pickle', 'rb'))
    n_sim = 1000
    metrics = []
    for i in range(n_sim):
        metrics.append(problem.simulate(
            results_NDPI['x'], results_NDPI['u'],
            results_NDPI['R'], results_NDPI['policy']
        ))
    ave_metrics = list(np.average(metrics, axis=0))
    file = open(f"results/{name}-sim.txt", "w+")
    file.write(f"Total output: {ave_metrics[0]}\n")
    file.write(f"Resource used/allocated: {ave_metrics[1]}/{ave_metrics[2]}\n")
    file.write(f"Dependency loss: {ave_metrics[3]}\n")
    file.write(f"Defended/total attacks: {ave_metrics[4]}/{ave_metrics[5]}\n")
    file.close()
    return


if __name__ == "__main__":
    main()
