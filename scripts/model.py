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
- Resource allocation + defense strategy
- LP + CMDP -> NLP
-------------------
"""

# import
import time
import logging
import numpy as np
import gurobipy as grb
import itertools as itl


# problem class
class CIS_Defense:
    """
    Initialization:
    `name`: name of instance;
    `G`: network defined by networkx, including resource cost `c_r(i)`,
        facility output `r(i)`, and service cost `c_s(i, j)`;
    `B`: total budget;
    `Pr(i)`: attack probability;
    `I_a`: list of attack intensities;
    `I_d`: list of defense intensities;
    `gamma`: discount factor
    """
    # initialization
    def __init__(self, name, H, G, Pr, I_a, I_d, gamma, runtime=3600):
        super().__init__()
        # logging
        logging.basicConfig(
            filename='LOG.log', filemode='w+', level=logging.INFO,
            format="%(levelname)s | %(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info("Instance {}...".format(name[-1]))
        self.name = name
        self.H = {
            h: [
                i for i in G.nodes() if G.nodes[i]['h'] == h
            ] for h in H.keys()
        }
        self.G = G
        self.Pr = Pr
        self.I_a = I_a
        self.I_d = I_d
        self.gamma = gamma
        self.effectiveness = 1
        # states
        self.S_lst = list(itl.product(range(2), repeat=len(self.G.nodes)))
        self.S = range(len(self.S_lst))
        # actions
        self.A_lst = list(itl.product([0]+self.I_d, repeat=len(self.G.nodes)))
        self.A = self.__filter_actions()
        # transition probability, only where T != 0: T[s, a][s_n] = pr
        self.T, self.T_rev = self.__generate_T()
        # state structure
        self.structure = {}
        for s in self.S:
            self.structure[s] = []
            for s_n in self.S:
                if s_n == s:
                    continue
                keep = True
                # states less than s
                for i in self.G.nodes.keys():
                    if self.S_lst[s][i] < self.S_lst[s_n][i]:
                        keep = False
                if keep:
                    self.structure[s].append(s)
        # big M
        self.M_theta = 1e6
        self.M_u = 1000
        self.M_sigma = 1e5
        self.M_lambda = 1000
        # time limit
        self.time_limit = runtime
        # initial distribution
        self.alpha = [1 / len(self.S)] * len(self.S)
        # independent set
        self.V_I = [
            i for i in self.G.nodes() if np.sum([
                self.G.nodes[i]['d'][key]
                for key in self.G.nodes[i]['d'].keys()
            ]) == 0
        ]
        # dependent set
        self.V_D = [
            i for i in self.G.nodes() if np.sum([
                self.G.nodes[i]['d'][key]
                for key in self.G.nodes[i]['d'].keys()
            ]) > 0
        ]

    def __filter_actions(self):
        """
        filter out actions
        """
        A = {}
        for s in self.S:
            A[s] = []
            for a in range(len(self.A_lst)):
                if np.min(
                    100 * np.array(self.S_lst[s]) - np.array(self.A_lst[a])
                ) >= 0:
                    A[s].append(a)
        return A

    def __contest_func(self, attack, defend):
        """
        The contest function between attacker & defender;
        Return the success pr of the attacker
        """
        return attack / (attack + self.effectiveness * defend)
        # return 0

    def __generate_T(self):
        """
        Generate T
        """
        # adjust attack probability
        Pr_s = {}
        for s in self.S:
            if s == 0:
                Pr_s[s] = np.array(
                    [1 / len(self.S_lst[s])] * len(self.S_lst[s])
                )
            else:
                Pr_s[s] = np.array([
                    self.S_lst[s][i] * self.Pr[i]
                    for i in self.G.nodes()
                ])
                Pr_s[s] = Pr_s[s] / np.sum(Pr_s[s])
        # calculate T, only T != 0; T_rev: states that transitions to s
        T, T_rev = {}, {s: set([]) for s in self.S}
        for s in self.S:
            for a in self.A[s]:
                # non-zero transitions
                T[s, a] = {}
                # absorbing state
                if s == 0:
                    T[s, a][s] = 1
                    T_rev[s].add(s)
                    continue
                # attack fails
                T[s, a][s] = float(np.sum([
                    Pr_s[s][i] * np.sum([
                        self.I_a[beta] * (
                            1 - self.__contest_func(beta, self.A_lst[a][i])
                        ) for beta in self.I_a.keys()
                    ]) for i in self.G.nodes()
                ]))
                T_rev[s].add(s)
                # attack succeeds
                for i in self.G.nodes():
                    # no transition
                    if self.S_lst[s][i] == 0:
                        continue
                    # transition
                    else:
                        # new s
                        s_n = self.S_lst.index(tuple([
                            self.S_lst[s][n] if n != i else 0
                            for n in self.G.nodes()
                        ]))
                        T[s, a][s_n] = float(Pr_s[s][i] * np.sum([
                            self.I_a[beta] * self.__contest_func(
                                beta, self.A_lst[a][i]
                            ) for beta in self.I_a.keys()
                        ]))
                        T_rev[s_n].add(s)
        return T, T_rev

    def __calculate_MDP_value(self, R, policy):
        """
        Calculate MDP value
        """
        # lhs
        lhs = np.zeros((len(self.S), len(self.S)))
        # update
        for s in self.S:
            for s_n in self.T[s, policy[s]]:
                if s == s_n:
                    lhs[s, s_n] = 1 - self.gamma * self.T[s, policy[s]][s_n]
                else:
                    lhs[s, s_n] = -1 * self.gamma * self.T[s, policy[s]][s_n]
        rhs = np.array([R[s].X for s in self.S])
        value = np.linalg.inv(lhs).dot(rhs)
        # return value
        return {s: value[s] for s in self.S}

    def __initialize(self):
        """
        clear and re-define
        """
        # models
        self.MP, self.DP = float("NaN"), float("NaN")
        self.mp, self.mp_cuts, self.dp_obj = float("NaN"), {}, {}
        self.CMDP_constr, self.CMDP_lhs = {}, {}
        # variables
        self.var_x, self.var_u, self.var_y, self.var_z = {}, {}, {}, {}
        self.var_r, self.var_sigma, self.var_delta = {}, {}, {}
        self.var_v, self.var_lambda, self.var_theta = {}, {}, float("NaN")
        self.var_nu, self.var_psi, self.var_rho, self.pi = {}, {}, {}, {}
        # values
        self.val_x, self.val_r, self.val_y = {}, {}, {}
        return

    # NLP
    def NLP(self, file_dir='None'):
        """
        Solve with NLP
        """
        logging.info("Solving NLP...")
        runtime = time.time()
        # model
        model = grb.Model()
        model.setParam("LogToConsole", 0)
        model.setParam("LogFile", "LOG.log")
        # model.setParam("DualReductions", 0)
        # model.setParam("IntFeasTol", 1e-5)
        # model.setParam("MIPFocus", 3)
        # model.setParam("NumericFocus", 0)
        # model.setParam("Presolve", -1)
        # model.setParam("Heuristics", 0.05)
        model.setParam("NonConvex", 2)
        model.setParam("TimeLimit", self.time_limit)
        # variables
        var_x, var_u, var_y, var_z = {}, {}, {}, {}
        var_r, var_sigma, var_delta = {}, {}, {}
        # all nodes
        for i in self.G.nodes():
            # x, amount of resource at n
            var_x[i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.INTEGER,
                name="x_{}".format(i)
            )
        # neighbor of independent nodes
        for edge in self.G.edges():
            # u, CIS connection
            var_u[edge] = model.addVar(
                lb=0, ub=1,
                vtype=grb.GRB.BINARY,
                name="u_{}_{}".format(edge[0], edge[1])
            )
            # z, CIS service
            var_z[edge] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="z_{}_{}".format(edge[0], edge[1])
            )
        # (s, a) pair
        for s in self.S:
            for a in self.A[s]:
                # y, occupancy measure
                var_y[s, a] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name="y_{}_{}".format(s, a)
                )
            # r, reward, not dependent on action
            var_r[s] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="r_{}".format(s)
            )
            for i in self.V_D:
                for h in self.H.keys():
                    # above threshold or not
                    var_sigma[s, i] = model.addVar(
                        lb=0, ub=1,
                        vtype=grb.GRB.BINARY,
                        name="sigma_{}_{}".format(s, i)
                    )
                # percentage output
                var_delta[s, i] = model.addVar(
                    lb=0, ub=self.G.nodes[i]['r'],
                    vtype=grb.GRB.CONTINUOUS,
                    name="delta_{}_{}".format(s, i)
                )
        model.update()
        # obj
        obj = grb.quicksum([
            # service cost
            -1 * grb.quicksum([
                self.G.edges[edge]['c_s'] * var_z[edge]
                for edge in self.G.edges()
            ]),
            # service cost
            -1 * grb.quicksum([
                self.G.edges[edge]['c_f'] * var_u[edge]
                for edge in self.G.edges()
            ]),
            # resource cost
            -1 * grb.quicksum([
                self.G.nodes[i]['c_r'] * var_x[i]
                for i in self.G.nodes.keys()
            ]),
            # CMDP value
            grb.quicksum([
                var_r[s] * var_y[s, a]
                for s in self.S
                for a in self.A[s]
            ])
        ])
        # set obj
        model.setObjective(obj, grb.GRB.MAXIMIZE)
        # constraints
        # connection & service
        for edge in self.G.edges():
            model.addLConstr(
                lhs=var_z[edge],
                sense=grb.GRB.LESS_EQUAL,
                rhs=self.M_u * var_u[edge]
            )
        # demand
        for j in self.G.nodes():
            for h in self.H.keys():
                model.addLConstr(
                    lhs=grb.quicksum([
                        var_z[i, j]
                        for i in self.H[h]
                        if i in self.G.predecessors(j)
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=self.G.nodes[j]['d'][h]
                )
        # supply
        for i in self.G.nodes():
            model.addLConstr(
                lhs=grb.quicksum([
                    var_z[i, j]
                    for j in self.G.successors(i)
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=self.G.nodes[i]['C']
            )
        # reward
        for s in self.S:
            # for a in self.A[s]:
            model.addLConstr(
                lhs=var_r[s],
                sense=grb.GRB.EQUAL,
                rhs=grb.quicksum([
                    # independent output
                    grb.quicksum([
                        self.S_lst[s][i] * self.G.nodes[i]['r']
                        for i in self.V_I
                    ]),
                    # dependent output
                    grb.quicksum([
                        self.S_lst[s][i] * var_delta[s, i]
                        for i in self.V_D
                    ])
                ])
            )
            # connection
            for s in self.S:
                for i in self.V_D:
                    # total demand at i
                    d_sum = np.sum([
                        self.G.nodes[i]['d'][h] for h in self.H.keys()
                    ])
                    # percentage output
                    model.addLConstr(
                        lhs=var_delta[s, i],
                        sense=grb.GRB.LESS_EQUAL,
                        rhs=(1 / d_sum) * grb.quicksum([
                            self.S_lst[s][j] * var_z[j, i]
                            for h in self.H.keys()
                            for j in self.H[h] if j in self.G.predecessors(i)
                        ]) * self.G.nodes[i]['r']
                    )
                    # sigma
                    for h in self.H.keys():
                        # no output below percentage
                        model.addLConstr(
                            lhs=var_delta[s, i],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=self.M_sigma * var_sigma[s, i]
                        )
                        if self.G.nodes[i]['d'][h] == 0:
                            continue
                        model.addLConstr(
                            lhs=self.G.nodes[i]['tau'][h] * var_sigma[s, i],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=(1 / self.G.nodes[i]['d'][h]) * grb.quicksum([
                                self.S_lst[s][j] * var_z[j, i]
                                for j in self.H[h]
                                if j in self.G.predecessors(i)
                            ])
                        )
        # CMDP
        for s in self.S:
            model.addLConstr(
                lhs=grb.quicksum([
                    grb.quicksum([
                        var_y[s, a]
                        for a in self.A[s]
                    ]),
                    -1 * self.gamma * grb.quicksum([
                        self.T[s_o, a][s] * var_y[s_o, a]
                        for s_o in self.T_rev[s]
                        for a in self.A[s_o]
                    ])
                ]),
                sense=grb.GRB.EQUAL,
                rhs=self.alpha[s],
                name="MDP_{}".format(s)
            )
        # resource constraints
        for i in self.G.nodes():
            model.addLConstr(
                lhs=grb.quicksum([
                    self.A_lst[a][i] * var_y[s, a]
                    for s in self.S
                    for a in self.A[s]
                ]),
                sense=grb.GRB.EQUAL,
                rhs=var_x[i],
                name="resource_{}".format(i)
            )
        model.update()
        # optimize
        model.optimize()
        runtime = time.time() - runtime
        # output
        if file_dir == "None":
            return
        file = open("{}/{}-NLP.txt".format(file_dir, self.name), "w+")
        # optimal value and gap
        file.write("==============================\n")
        # no obj
        try:
            # CMDP obj
            CMDP_obj = np.sum([
                model.getVarByName("r_{}".format(s)).X
                * model.getVarByName("y_{}_{}".format(s, a)).X
                for s in self.S
                for a in self.A[s]
            ])
            # obj
            file.write("Objective: {};\n".format(model.ObjVal))
            # gap
            file.write("Gap: {};\n".format(model.MIPGap))
            file.write("==============================\n")
            # time
            file.write(
                "Solving time: {} seconds;\n".format(runtime)
            )
            # first stage results
            file.write("==============================\n")
            # planning cost
            resource_cost = np.sum([
                self.G.nodes[i]['c_r'] * model.getVarByName("x_{}".format(i)).X
                for i in self.G.nodes()
            ])
            connection_cost = np.sum([
                self.G.edges[edge]['c_f'] * model.getVarByName(
                    "u_{}_{}".format(edge[0], edge[1])
                ).X for edge in self.G.edges()
            ])
            service_cost = np.sum([
                self.G.edges[edge]['c_s'] * model.getVarByName(
                    "z_{}_{}".format(edge[0], edge[1])
                ).X for edge in self.G.edges()
            ])
            file.write("Total planning cost: {}\n".format(np.sum([
                resource_cost, connection_cost
            ])))
            file.write("    Resource cost: {}\n".format(resource_cost))
            file.write("    Connection cost: {}\n".format(connection_cost))
            file.write("    Service cost: {}\n".format(service_cost))
            file.write("------------------------------\n")
            file.write("Resource allocation:\n")
            for i in self.G.nodes.keys():
                file.write("    At {}: {}\n".format(
                    i, model.getVarByName("x_{}".format(i)).X)
                )
            file.write("Service:\n")
            for edge in self.G.edges():
                if model.getVarByName(
                    "u_{}_{}".format(edge[0], edge[1])
                ).X > 0.9:
                    file.write(
                        "    ({}, {}) costs - c^f: {}, c^s: {}\n".format(
                            edge[0], edge[1], self.G.edges[edge]['c_f'],
                            self.G.edges[edge]['c_s'] * model.getVarByName(
                                "z_{}_{}".format(edge[0], edge[1])
                            ).X
                        )
                    )
            # MDP values
            file.write("==============================\n")
            file.write("CMDP obj: {}\n".format(CMDP_obj))
            file.write("------------------------------\n")
            # policy
            policy = {}
            for s in self.S:
                opt_a = self.A[s][np.argmax([
                    model.getVarByName("y_{}_{}".format(s, a)).X
                    for a in self.A[s]
                ])]
                policy[s] = opt_a
            # calculate value
            value = self.__calculate_MDP_value(var_r, policy)
            # write all values
            for s in self.S:
                file.write("    State: {}, value {}\n".format(
                    self.S_lst[s], value[s]
                ))
                file.write("    State: {}, r {}\n".format(
                    self.S_lst[s], var_r[s].X
                ))
            file.write("------------------------------\n")
            # write policy
            for s in self.S:
                file.write("    State: {}, policy: {}\n".format(
                    self.S_lst[s], self.A_lst[policy[s]]
                ))
            file.write("------------------------------\n")
            # constraints
            file.write("Resource usage: \n")
            for i in self.G.nodes.keys():
                file.write(
                    "    Usage at {}: {:.2f}, available: {:.2f}\n".format(
                        i, np.sum([
                            self.A_lst[a][i]
                            * model.getVarByName("y_{}_{}".format(s, a)).X
                            for s in self.S
                            for a in self.A[s]
                        ]), model.getVarByName("x_{}".format(i)).X
                    )
                )
            file.write("==============================\n")
        except AttributeError:
            file.write("Optimal status {};\n".format(model.status))
            return {
                'obj': float('nan'), 'runtime': runtime,
                'gap': float('nan')
            }
        return {'obj': model.ObjVal, 'runtime': runtime, 'gap': model.MIPGap}

    # GBD/NDPI - MP
    def __build_MP(self):
        """Build MP for Bedners"""
        # model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        # model.setParam("DualReductions", 0)
        # model.setParam("IntFeasTol", 1e-5)
        # model.setParam("MIPFocus", 3)
        # model.setParam("Heuristics", 0)
        # model.setParam("NumericFocus", 0)
        # model.setParam("Presolve", -1)
        # model.setParam("NonConvex", -1)
        model.setParam("TimeLimit", self.time_limit)
        # all nodes
        for i in self.G.nodes():
            # x, amount of resource at n
            self.var_x[i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.INTEGER,
                name="x_{}".format(i)
            )
        # neighbor of independent nodes
        for edge in self.G.edges():
            # u, CIS connection
            self.var_u[edge] = model.addVar(
                lb=0, ub=1,
                vtype=grb.GRB.BINARY,
                name="u_{}_{}".format(edge[0], edge[1])
            )
            # z, CIS service
            self.var_z[edge] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="z_{}_{}".format(edge[0], edge[1])
            )
        # s
        for s in self.S:
            # r, reward, not dependent on action
            self.var_r[s] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="r_{}".format(s)
            )
            for i in self.V_D:
                for h in self.H.keys():
                    # above threshold or not
                    self.var_sigma[s, i] = model.addVar(
                        lb=0, ub=1,
                        vtype=grb.GRB.BINARY,
                        name="sigma_{}_{}".format(s, i)
                    )
                # percentage output
                self.var_delta[s, i] = model.addVar(
                    lb=0, ub=self.G.nodes[i]['r'],
                    vtype=grb.GRB.CONTINUOUS,
                    name="delta_{}_{}".format(s, i)
                )
        # theta
        self.var_theta = model.addVar(
            lb=-grb.GRB.INFINITY, ub=self.M_theta,
            vtype=grb.GRB.CONTINUOUS, name="theta"
        )
        model.update()
        # obj
        obj = grb.quicksum([
            # service cost
            -1 * grb.quicksum([
                self.G.edges[edge]['c_s'] * self.var_z[edge]
                for edge in self.G.edges()
            ]),
            # service cost
            -1 * grb.quicksum([
                self.G.edges[edge]['c_f'] * self.var_u[edge]
                for edge in self.G.edges()
            ]),
            # resource cost
            -1 * grb.quicksum([
                self.G.nodes[i]['c_r'] * self.var_x[i]
                for i in self.G.nodes()
            ]),
            # CMDP value
            self.var_theta
        ])
        # set obj
        model.setObjective(obj, grb.GRB.MAXIMIZE)
        # constraints
        # connection & service
        for edge in self.G.edges():
            model.addLConstr(
                lhs=self.var_z[edge],
                sense=grb.GRB.LESS_EQUAL,
                rhs=self.M_u * self.var_u[edge]
            )
        # demand
        for j in self.G.nodes():
            for h in self.H.keys():
                model.addLConstr(
                    lhs=grb.quicksum([
                        self.var_z[i, j]
                        for i in self.H[h]
                        if i in self.G.predecessors(j)
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=self.G.nodes[j]['d'][h]
                )
        # supply
        for i in self.G.nodes():
            model.addLConstr(
                lhs=grb.quicksum([
                    self.var_z[i, j]
                    for j in self.G.successors(i)
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=self.G.nodes[i]['C']
            )
        # reward
        for s in self.S:
            # for a in self.A[s]:
            model.addLConstr(
                lhs=self.var_r[s],
                sense=grb.GRB.EQUAL,
                rhs=grb.quicksum([
                    # independent output
                    grb.quicksum([
                        self.S_lst[s][i] * self.G.nodes[i]['r']
                        for i in self.V_I
                    ]),
                    # dependent output
                    grb.quicksum([
                        self.S_lst[s][i] * self.var_delta[s, i]
                        for i in self.V_D
                    ])
                ])
            )
            # connection
            for i in self.V_D:
                # total demand at i
                d_sum = np.sum([
                    self.G.nodes[i]['d'][h] for h in self.H.keys()
                ])
                # percentage output
                model.addLConstr(
                    lhs=self.var_delta[s, i],
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=(1 / d_sum) * grb.quicksum([
                        self.S_lst[s][j] * self.var_z[j, i]
                        for h in self.H.keys()
                        for j in self.H[h] if j in self.G.predecessors(i)
                    ]) * self.G.nodes[i]['r']
                )
                # sigma
                for h in self.H.keys():
                    # no output below percentage
                    model.addLConstr(
                        lhs=self.var_delta[s, i],
                        sense=grb.GRB.LESS_EQUAL,
                        rhs=self.M_sigma * self.var_sigma[s, i]
                    )
                    if self.G.nodes[i]['d'][h] == 0:
                        continue
                    model.addLConstr(
                        lhs=self.G.nodes[i]['tau'][h]
                        * self.var_sigma[s, i],
                        sense=grb.GRB.LESS_EQUAL,
                        rhs=(1 / self.G.nodes[i]['d'][h]) * grb.quicksum([
                            self.S_lst[s][j] * self.var_z[j, i]
                            for j in self.H[h]
                            if j in self.G.predecessors(i)
                        ])
                    )
        model.update()
        return model

    # GBD - DP
    def __build_DP(self):
        """Build DP as CMDP"""
        # model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        # model.setParam("IntFeasTol", 1e-5)
        # model.setParam("NumericFocus", 0)
        model.setParam("DualReductions", 0)
        # value, v
        for s in self.S:
            self.var_v[s] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="v_{}".format(s)
            )
        model.update()
        # lambda, additional constraints
        for i in self.G.nodes():
            self.var_lambda[i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="lambda_{}".format(i)
            )
        model.update()
        # objective
        objective = grb.quicksum([
            grb.quicksum([
                self.alpha[s] * self.var_v[s]
                for s in self.S
            ]),
            grb.quicksum([
                self.var_x[i].X * self.var_lambda[i]
                for i in self.G.nodes()
            ]),
        ])
        model.setObjective(objective, grb.GRB.MINIMIZE)
        # CMDP constraint
        for s in self.S:
            for a in self.A[s]:
                self.CMDP_lhs[s, a] = grb.quicksum([
                    self.var_v[s],
                    -1 * grb.quicksum([
                        self.gamma * self.T[s, a][s_n] * self.var_v[s_n]
                        for s_n in self.T[s, a].keys()
                    ]),
                    grb.quicksum([
                        self.A_lst[a][i] * self.var_lambda[i]
                        for i in self.G.nodes()
                    ]),
                ])
                self.CMDP_constr[s, a] = model.addLConstr(
                    lhs=self.CMDP_lhs[s, a],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=self.var_r[s].X
                )
        model.update()
        return model

    # Generalized Benders decomposition
    def GBD(self, epsilon=0.001, file_dir='None'):
        """
        Benders decomposition, two stages.
        MP: planning phase;
        SP: CMDP
        """
        self.__initialize()
        logging.info("Solving GBD...")
        runtime = time.time()
        runtime_dual = 0
        # MP
        self.MP = self.__build_MP()
        # start iteration
        iteration, optimal = 0, False
        while not optimal:
            # time limit
            if time.time() - runtime > self.time_limit:
                break
            # solve MP
            self.MP.optimize()
            # solve DP
            dual_start = time.time()
            self.DP = self.__build_DP()
            self.DP.optimize()
            runtime_dual += time.time() - dual_start
            logging.info(f"{iteration} - theta: {self.var_theta.X}, "
                         f"dual: {self.DP.ObjVal}")
            # condition
            if np.abs(
                self.var_theta.X - self.DP.ObjVal
            ) > epsilon and self.var_theta.X > self.DP.ObjVal:
                # calculate non zero y
                val_y, A_y = {}, {}
                for s in self.S:
                    A_y[s] = []
                    for a in self.A[s]:
                        if self.CMDP_constr[s, a].Pi != 0:
                            val_y[s, a] = self.CMDP_constr[s, a].Pi
                            A_y[s].append(a)
                self.MP.addLConstr(
                    lhs=self.var_theta,
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=grb.quicksum([
                        grb.quicksum([
                            val_y[s, a] * grb.quicksum([
                                self.var_r[s],
                                -1 * self.CMDP_lhs[s, a].getValue()
                            ])
                            for s in self.S
                            for a in A_y[s]
                        ]),
                        grb.quicksum([
                            self.alpha[s] * self.var_v[s].X
                            for s in self.S
                        ]),
                        grb.quicksum([
                            self.var_lambda[i].X * self.var_x[i]
                            for i in self.G.nodes()
                        ]),
                    ])
                )
            else:
                optimal = True
            iteration += 1
        # reconstruct solution
        for i in self.G.nodes():
            self.var_x[i].LB = np.sum([
                self.A_lst[a][i] * self.CMDP_constr[s, a].Pi
                for s in self.S for a in self.A[s]
            ])
            self.var_theta.Obj = 0.0
        for s in self.S:
            self.var_r[s].UB = self.var_r[s].X
            self.var_r[s].LB = self.var_r[s].X
        self.MP.update()
        self.MP.optimize()
        # time
        runtime = time.time() - runtime
        # print solution to file
        if file_dir == "None":
            return
        file = open("{}/{}-GBD.txt".format(file_dir, self.name), "w+")
        file.write("==============================\n")
        # not optimal
        try:
            # obj
            obj = self.MP.ObjVal + self.DP.ObjVal
            file.write("Objective: {};\n".format(obj))
        except AttributeError:
            file.write("Optimal status {};\n".format(self.MP.status))
            return {
                'obj': float('nan'), 'runtime': runtime,
                'gap': float('nan')
            }
        # gap
        file.write("Gap: {};\n".format(self.MP.MIPGap))
        file.write("==============================\n")
        # time
        file.write(
            "Solving time: {} seconds;\n".format(runtime)
        )
        # first stage results
        file.write("==============================\n")
        # planning cost
        resource_cost = np.sum([
            self.G.nodes[i]['c_r'] * self.var_x[i].X
            for i in self.G.nodes()
        ])
        connection_cost = np.sum([
            self.G.edges[edge]['c_f'] * self.var_u[edge].X
            for edge in self.G.edges()
        ])
        service_cost = np.sum([
            self.G.edges[edge]['c_s'] * self.var_z[edge].X
            for edge in self.G.edges()
        ])
        file.write("Total planning cost: {}\n".format(np.sum([
            resource_cost, connection_cost
        ])))
        file.write("    Resource cost: {}\n".format(resource_cost))
        file.write("    Connection cost: {}\n".format(connection_cost))
        file.write("    Service cost: {}\n".format(service_cost))
        file.write("------------------------------\n")
        file.write("Resource allocation:\n")
        for i in self.G.nodes():
            file.write("    At {}: {}\n".format(i, self.var_x[i].X))
        file.write("Service:\n")
        for edge in self.G.edges():
            if self.var_u[edge].X > 0.9:
                file.write("    ({}, {}) costs - c^f: {}, c^s: {}\n".format(
                    edge[0], edge[1], self.G.edges[edge]['c_f'],
                    self.G.edges[edge]['c_s'] * self.var_z[edge].X
                ))
        # CMDP values
        file.write("==============================\n")
        file.write("CMDP obj: {}\n".format(self.DP.ObjVal))
        file.write("------------------------------\n")
        for s in self.S:
            file.write("    State: {}, value: {}\n".format(
                self.S_lst[s], self.var_v[s].X
            ))
            file.write("    State: {}, r: {}\n".format(
                self.S_lst[s], self.var_r[s].X
            ))
        file.write("------------------------------\n")
        # policy
        policy = {}
        for s in self.S:
            opt_a = self.A[s][np.argmax([
                self.CMDP_constr[s, a].Pi for a in self.A[s]
            ])]
            policy[s] = opt_a
        # write policy
        for s in self.S:
            file.write("    State: {}, policy: {}\n".format(
                self.S_lst[s], self.A_lst[policy[s]]
            ))
        file.write("------------------------------\n")
        # constraints
        file.write("Resource usage: \n")
        for i in self.G.nodes():
            file.write("    Usage at {}: {:.2f}, available: {:.2f}\n".format(
                i, np.sum([
                    self.A_lst[a][i] * self.CMDP_constr[s, a].Pi
                    for s in self.S
                    for a in self.A[s]
                ]), self.var_x[i].X
            ))
        file.close()
        # return
        return {
            'obj': self.MP.ObjVal, 'runtime': runtime, 'gap': self.MP.MIPGap
        }

    # NDPI mp
    def __build_mp(self):
        """
        NDPI build mp and dp obj
        """
        # mp
        model = grb.Model()
        model.setParam("OutputFlag", False)
        # nu
        for s in self.S:
            self.var_nu[s] = model.addVar(
                lb=-self.M_theta, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="nu_{}".format(s)
            )
        # rho
        for i in self.G.nodes():
            self.var_rho[i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="rho_{}".format(i)
            )
        objective = grb.quicksum([
            grb.quicksum([
                self.alpha[s] * self.var_nu[s]
                for s in self.S
            ]),
            grb.quicksum([
                self.var_rho[i] * self.M_lambda
                for i in self.G.nodes.keys()
            ])
        ])
        model.setObjective(objective, grb.GRB.MINIMIZE)
        # obj for dp
        for s in self.S:
            # psi
            self.var_psi[s] = model.addVars(
                self.A[s], lb=-float('inf'), ub=float('inf')
            )
            for a in self.A[s]:
                self.dp_obj[s, a] = grb.quicksum([
                    self.gamma * grb.quicksum([
                        self.T[s, a][s_n] * self.var_nu[s_n]
                        for s_n in self.T[s, a].keys()
                    ]),
                    -1 * grb.quicksum([
                        self.var_rho[i] * self.A_lst[a][i]
                        for i in self.G.nodes.keys()
                    ])
                ])
                model.addLConstr(
                    lhs=self.var_psi[s].select(a)[0],
                    sense=grb.GRB.EQUAL,
                    rhs=self.dp_obj[s, a]
                )
        # structure
        for s in self.S:
            for s_n in self.structure[s]:
                model.addLConstr(
                    lhs=self.var_nu[s],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=self.var_nu[s_n]
                )
        model.update()
        return model

    # NDPI retrieve solution
    def __retrieve_solution(self, model):
        """
        retrieve solution
        """
        # x
        for i in self.G.nodes():
            self.val_x[i] = model.cbGetSolution(self.var_x[i])
        # r
        for s in self.S:
            self.val_r[s] = model.cbGetSolution(self.var_r[s])
        return

    # PDPI
    def __PDPI(self, epsilon):
        """
        Implement PDPI for CMDP
        """
        # update mp
        for i in self.G.nodes():
            self.var_rho[i].Obj = self.val_x[i]
        for cut in self.mp_cuts.values():
            self.mp.remove(cut)
        self.mp.reset(0)
        # reset
        self.pi, self.mp_cuts = {}, {}
        # loop
        k, optimal = 0, False
        while not optimal:
            # safeguard
            if k >= 100:
                break
            # solve MP
            self.mp.optimize()
            # flag
            optimal = True
            # loop states
            for s in self.S:
                # update dp
                q = np.sum([
                    [self.val_r[s]] * len(self.A[s]),
                    self.mp.getAttr("X", self.var_psi[s]).select()
                ], axis=0)
                q_ind = np.argmax(q)
                self.pi[s] = self.A[s][q_ind]
                # comapre state value
                if np.abs(self.var_nu[s].X - q[q_ind]) > epsilon:
                    optimal = False
                    # PDPI cut
                    if (s, self.pi[s]) in self.mp_cuts.keys():
                        continue
                    self.mp_cuts[s, self.pi[s]] = self.mp.addLConstr(
                        lhs=self.var_nu[s],
                        sense=grb.GRB.GREATER_EQUAL,
                        rhs=grb.quicksum([
                            self.val_r[s],
                            self.var_psi[s].select(self.pi[s])[0]
                        ])
                    )
            # next iteration
            k += 1
        # obtain y
        self.val_y = {}
        for key in self.mp_cuts.keys():
            if self.mp_cuts[key].Pi != 0:
                self.val_y[key] = self.mp_cuts[key].Pi
        return

    # NDPI call back
    def __callback(self, model, where, epsilon):
        """
        Call back for NDPI
        """
        # incumbent integer solutions
        if where == grb.GRB.Callback.MIPSOL:
            callback_time = time.time()
            # retrieve solution
            self.__retrieve_solution(model)
            self.__PDPI(epsilon)
            # add GBD cut
            self.MP.cbLazy(
                lhs=self.var_theta,
                sense=grb.GRB.LESS_EQUAL,
                rhs=grb.quicksum([
                    grb.quicksum([
                        self.val_y[key] * grb.quicksum([
                            self.var_r[key[0]],
                            -1 * self.var_nu[key[0]].X,
                            self.var_psi[key[0]].select(key[1])[0].X
                        ])
                        for key in self.val_y.keys()
                    ]),
                    grb.quicksum([
                        self.alpha[s] * self.var_nu[s].X
                        for s in self.S
                    ]),
                    grb.quicksum([
                        self.var_rho[i].X * self.var_x[i]
                        for i in self.G.nodes()
                    ])
                ])
            )
            self.callback_time += time.time() - callback_time
        return

    # NDPI
    def NDPI(self, epsilon=0.001, file_dir='None'):
        """
        Benders decomposition, two stages.
        MP: planning phase;
        SP: CMDP
        """
        logging.info("Solving NDPI...")
        self.__initialize()
        runtime = time.time()
        # MP & mp
        self.MP = self.__build_MP()
        self.mp = self.__build_mp()
        self.callback_time = 0
        # B&B with cuts
        self.MP.setParam("OutputFlag", True)
        self.MP.setParam("LogToConsole", 0)
        self.MP.setParam("LogFile", "LOG.log")
        self.MP.setParam("PreCrush", 1)
        self.MP.setParam("LazyConstraints", 1)
        # self.MP.setParam("MIPFocus", 2)
        # self.MP.setParam("Heuristics", 0.00)
        self.MP.optimize(
            lambda model, where: self.__callback(model, where, epsilon)
        )
        gap = self.MP.MIPGap
        # reconstruct solution
        for i in self.G.nodes():
            self.var_x[i].LB = np.sum([
                self.A_lst[key[1]][i] * self.val_y[key]
                for key in self.val_y.keys()
            ])
            self.var_theta.Obj = 0.0
        for s in self.S:
            self.var_r[s].UB = self.var_r[s].X
            self.var_r[s].LB = self.var_r[s].X
        self.MP.optimize()
        # time
        runtime = time.time() - runtime
        # print solution to file
        if file_dir == "None":
            return
        file = open("{}/{}-NDPI.txt".format(file_dir, self.name), "w+")
        file.write("==============================\n")
        # not optimal
        try:
            # obj
            obj = self.MP.ObjVal + self.mp.ObjVal
            file.write("Objective: {};\n".format(obj))
        except AttributeError:
            file.write("Optimal status {};\n".format(self.MP.status))
            return {
                'obj': float('nan'), 'runtime': runtime, 'gap': float('nan')
            }
        # gap
        file.write("Gap: {};\n".format(gap))
        file.write("==============================\n")
        # time
        file.write(f"Solving time: {runtime} seconds;\n")
        file.write(f"Calback time: {self.callback_time} seconds;\n")
        # first stage results
        file.write("==============================\n")
        # planning cost
        resource_cost = np.sum([
            self.G.nodes[i]['c_r'] * self.var_x[i].X
            for i in self.G.nodes()
        ])
        connection_cost = np.sum([
            self.G.edges[edge]['c_f'] * self.var_u[edge].X
            for edge in self.G.edges()
        ])
        service_cost = np.sum([
            self.G.edges[edge]['c_s'] * self.var_z[edge].X
            for edge in self.G.edges()
        ])
        file.write("Total planning cost: {}\n".format(np.sum([
            resource_cost, connection_cost
        ])))
        file.write("    Resource cost: {}\n".format(resource_cost))
        file.write("    Connection cost: {}\n".format(connection_cost))
        file.write("    Service cost: {}\n".format(service_cost))
        file.write("------------------------------\n")
        file.write("Resource allocation:\n")
        for i in self.G.nodes():
            file.write("    At {}: {}\n".format(i, self.var_x[i].X))
        file.write("Service:\n")
        for edge in self.G.edges():
            if self.var_u[edge].X > 0.9:
                file.write("    ({}, {}) costs - c^f: {}, c^s: {}\n".format(
                    edge[0], edge[1], self.G.edges[edge]['c_f'],
                    self.G.edges[edge]['c_s'] * self.var_z[edge].X
                ))
        # CMDP values
        file.write("==============================\n")
        file.write("CMDP obj: {}\n".format(self.mp.ObjVal))
        file.write("------------------------------\n")
        for s in self.S:
            file.write("    State: {}, value: {}\n".format(
                self.S_lst[s], self.var_nu[s].X
            ))
            file.write("    State: {}, r: {}\n".format(
                self.S_lst[s], self.var_r[s].X
            ))
        file.write("------------------------------\n")
        # policy
        policy = {}
        for s in self.S:
            policy[s] = self.pi[s]
        # write policy
        for s in self.S:
            file.write("    State: {}, policy: {}\n".format(
                self.S_lst[s], self.A_lst[policy[s]]
            ))
        file.write("------------------------------\n")
        # constraints
        file.write("Resource usage: \n")
        for i in self.G.nodes():
            file.write("    Usage at {}: {:.2f}, available: {:.2f}\n".format(
                i, np.sum([
                    self.A_lst[key[1]][i] * self.val_y[key]
                    for key in self.val_y.keys()
                ]), self.var_x[i].X
            ))
        file.close()
        # return
        return {
            'obj': self.MP.ObjVal, 'runtime': runtime, 'gap': self.MP.MIPGap
        }
