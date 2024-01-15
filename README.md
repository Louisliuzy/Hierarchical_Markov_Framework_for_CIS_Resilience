# A Two-Stage Optimization Framework for Enhancing Interconnected Critical Infrastructure Systems Resilience Under Sequential Attacks

This archive is distributed under the [MIT License](LICENSE).

The source code and data in this repository are a snapshot of the software and data
that were used in the research reported on in the manuscript under review 
[A Two-Stage Optimization Framework for Enhancing Interconnected Critical Infrastructure Systems Resilience Under Sequential Attacks](https://www.researchgate.net/publication/377220359_A_Two-Stage_Optimization_Framework_to_Enhance_Interconnected_Critical_Infrastructure_Systems_Resilience_Under_Sequential_Attacks) by Z. Liu et. al. The data generated for this study are included in the manuscript and the codes.

## Cite

To cite this repository, please cite the [manuscript](https://www.researchgate.net/publication/377220359_A_Two-Stage_Optimization_Framework_to_Enhance_Interconnected_Critical_Infrastructure_Systems_Resilience_Under_Sequential_Attacks).

Below is the BibTex for citing the manuscript.

```
@article{Liu2024,
  title={A Two-Stage Optimization Framework to Enhance Interconnected Critical Infrastructure Systems Resilience Under Sequential Attacks},
  author={Liu, Zeyu and Li, Xueping and Khojandi, Anahita},
  journal={Preprint},
  year={2024},
  url={https://www.researchgate.net/publication/377220359_A_Two-Stage_Optimization_Framework_to_Enhance_Interconnected_Critical_Infrastructure_Systems_Resilience_Under_Sequential_Attacks}
},
  doi={10.13140/RG.2.2.11406.54087}
```

## Description

The goal of this repository is to develop a novel framework that enhances interdependent CIS resilience through a combination of strategic and operational decision-making. We divide the decision-making horizon into two stages: the planning stage and the operational stage. In the planning stage, we model pre-attack network design and defense resource allocations. We use an integer program to optimize the strategic defense planning decisions, including the interconnectivity between CIS facilities and the allocation of defense resources. In the operational stage, we model defense strategies under an uncertain environment using a constrained Markov decision process (CMDP). The data used in this study contains randomly generated instances and a case study. Codes for randomly generating instances are provided in this repository. Data for the case study can be found in the manuscript.


## Python Prerequisite

The following Python libraries are required to run the source codes:
1. `numpy`;
2. `scipy`;
3. `pickle`;
4. `networkx`;
5. `gurobipy`.


## Usage

Run `CIS()` function in the `main.py` file in the `scripts/` folder. Different instances can be set up using the provided variables of network size, total budget, and attack and defense intensities. Three algorithms are available, including the default Gurobi solver (`NLP`), generalized Benders decomposition (`GBD`), and a novel nested decomposition with policy iteration (`NDPI`). Note that it may take hours to complete larger instances especially for `NLP`. Please refer to the manuscript for detailed algorithm comparison.


## Support

For support in using this software, submit an
[issue](https://github.com/Louisliuzy/Two-Stage_Optimization_for_CIS_Resilience/issues/new).
