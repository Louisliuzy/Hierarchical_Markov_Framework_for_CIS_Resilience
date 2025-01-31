# A Hierarchical Markov Decision Framework for Enhancing Critical Infrastructure Resilience Under Sequential Interdictions

This archive is distributed under the [MIT License](LICENSE).

The source code and data in this repository are a snapshot of the software and data
that were used in the research reported on in the manuscript under review: "A Hierarchical Markov Decision Framework for Enhancing Critical Infrastructure Resilience Under Sequential Interdictions" by Z. Liu et. al. The data generated for this study are included in the manuscript and the codes.

## Cite

To cite this repository, please cite the manuscript.

Below is the BibTex for citing the manuscript.

```
@article{Liu2024,
  title={A Hierarchical Markov Decision Framework for Enhancing Critical Infrastructure Resilience Under Sequential Interdictions},
  author={Liu, Zeyu and Li, Xueping and Khojandi, Anahita},
  journal={Preprint},
  year={2024},
}
```

## Description

This repository stores the data and scripts of a study that develops a novel framework to enhance the resilience of multi-level interdependent CIS through a combination of strategic and operational decision-making. We divide the decision-making horizon into the planning stage and the operational stage. In the planning stage, we model pre-attack network design and defense resource allocations. We use an integer program to optimize the strategic defense planning decisions, including the interconnectivity between CIS facilities and the allocation of defense resources. In the operational stage, we model defense strategies under an uncertain environment using a constrained Markov decision process (CMDP). A nested decomposition algorithm is designed and is integrated into a branch-and-cut method. The data used in this study contains randomly generated instances and real-world values. Scripts for randomly generating instances are provided in this repository. Data for the case study can be found in the manuscript.


## Python Prerequisite

The following Python libraries are required to run the source codes:
1. `numpy`;
2. `scipy`;
3. `pickle`;
4. `cartopy`;
5. `seaborn`;
6. `networkx`;
7. `gurobipy`;
8. `matplotlib`.


## Usage

Run `CIS()` function in the `main.py` file in the `scripts/` folder. Different instances can be set up using the provided variables of network size, total budget, and attack and defense intensities. Three algorithms are available, including the default Gurobi solver (`NLP`), generalized Benders decomposition (`GBD`), and a novel nested decomposition with policy iteration (`NDPI`). Plot functions are also available in the `main.py` file for generating results. In addition, run the `knox.py` file in the `scripts/` folder for the case study. Note that model runtime may be long for large instances. Please refer to the manuscript for detailed algorithm comparison. Plot functions are located in the `plot.py` file.


## Support

For support in using this software, submit an
[issue](https://github.com/Louisliuzy/Hierarchical_Markov_Framework_for_CIS_Resilience/issues).
