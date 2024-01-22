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
- Resource allocation + defense strategy
- LP + CMDP -> NLP
-------------------
"""

# import
import time
import pickle
import logging
import numpy as np
import gurobipy as grb
import itertools as itl


# problem class
class Knox_Defense:
    """
    Grid connection. Initialization:
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
    def __init__(self, name, G, B, Pr, I_a, I_d, gamma):
        super().__init__()
        # logging
        logging.basicConfig(
            filename='LOG.log', filemode='w+', level=logging.INFO,
            format="%(levelname)s | %(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info("Instance {}...".format(name))
        self.name = name
        self.G = G
        self.B = B
        self.Pr = Pr
        self.I_a = I_a
        self.I_d = I_d
        self.gamma = gamma
        # try loading S, A, and T
        try:
            saved_model = self.__load()
            logging.info("Loading model...")
            self.S_lst = saved_model["S_lst"]
            self.S = saved_model["S"]
            self.A_lst = saved_model["A_lst"]
            self.A = saved_model["A"]
            self.T = saved_model["T"]
        except FileNotFoundError:
            # states
            self.S_lst = list(itl.product(range(2), repeat=len(self.G.nodes)))
            self.S = range(len(self.S_lst))
            # actions
            self.A_lst = list(itl.product(
                [0] + self.I_d, repeat=len(self.G.nodes)
            ))
            logging.info("Filtering actions...")
            self.A = self.__filter_actions()
            logging.info("Generating T...")
            # transition probability, only where T != 0: T[s, a][s_n] = pr
            self.T, self.T_rev = self.__generate_T()
            # save model
            self.__save()
        # big M
        self.M = 1e6
        # time limit
        self.time_limit = 86400 * 4
        # initial distribution
        # self.alpha = [1 / len(self.S)] * len(self.S)
        self.alpha = [0] * (len(self.S) - 1) + [1]
        # independent set
        self.V_I = [
            i for i in self.G.nodes.keys() if self.G.nodes[i]['type'] == 'I'
        ]
        # dependent set
        self.V_D = [
            i for i in self.G.nodes.keys() if self.G.nodes[i]['type'] == 'D'
        ]
        # Benders MP and DP
        self.MP, self.DP = float("NaN"), float("NaN")
        self.CMDP_constr, self.CMDP_lhs = {}, {}
        # Benders variables
        self.var_x, self.var_u, self.var_r = {}, {}, {}
        self.var_sigma, self.var_theta = {}, float("NaN")
        self.var_v, self.var_lambda = {}, {}
        # final policy
        self.policy = float('nan')

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
        return attack / (attack + defend)
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
                    for i in self.G.nodes.keys()
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
                    ]) for i in self.G.nodes.keys()
                ]))
                T_rev[s].add(s)
                # attack succeeds
                for i in self.G.nodes.keys():
                    # no transition
                    if self.S_lst[s][i] == 0:
                        continue
                    # transition
                    else:
                        # new s
                        s_n = self.S_lst.index(tuple([
                            self.S_lst[s][n] if n != i else 0
                            for n in self.G.nodes.keys()
                        ]))
                        T[s, a][s_n] = float(Pr_s[s][i] * np.sum([
                            self.I_a[beta] * self.__contest_func(
                                beta, self.A_lst[a][i]
                            ) for beta in self.I_a.keys()
                        ]))
                        T_rev[s_n].add(s)
        return T, T_rev

    def __save(self):
        """save model"""
        pickle.dump({
            "name": self.name, "S_lst": self.S_lst, "S": self.S,
            "A_lst": self.A_lst, "A": self.A, "T": self.T, "T_rev": self.T_rev
        }, open(f'models/{self.name}.pickle', 'wb'))
        return

    def __load(self):
        """load model"""
        return pickle.load(open(f'models/{self.name}.pickle', 'rb'))

    # Generalized Benders decomposition
    def GBD(self, epsilon=0.001, file_dir='None'):
        """
        Benders decomposition, two stages.
        MP: planning phase;
        SP: CMDP
        """
        logging.info("Solving GBD...")
        runtime = time.time()
        runtime_dual = 0
        # MP
        self.MP = self.__build_MP()
        # DP
        self.DP = self.__build_DP()
        # start iteration
        iteration, optimal = 0, False
        while not optimal:
            # time limit
            if time.time() - runtime > self.time_limit:
                break
            if iteration > 51:
                break
            # solve MP
            self.MP.optimize()
            logging.info(f"{iteration} - {self.MP.ObjVal}")
            # modify DP and solve
            dual_start = time.time()
            self.__update_DP()
            # self.__build_DP_full()
            self.DP.optimize()
            runtime_dual += time.time() - dual_start
            # condition
            if np.abs(
                self.var_theta.X - self.DP.ObjVal
            ) > epsilon:
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
                                self.var_r[s, a],
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
                            for i in self.G.nodes.keys()
                        ])
                    ])
                )
            else:
                optimal = True
            iteration += 1
        # time
        runtime = time.time() - runtime
        # print solution to file
        if file_dir == "None":
            return
        file = open("{}/{}-GBD.txt".format(file_dir, self.name), "w+")
        file.write("==============================\n")
        # not optimal
        if self.MP.status != grb.GRB.OPTIMAL:
            file.write("Optimal status {};\n".format(self.MP.status))
            return {
                'obj': float('nan'), 'runtime': runtime,
                'gap': float('nan')
            }
        # calculate obj
        planning_cost = -(self.MP.ObjVal - self.var_theta.X)
        obj = self.DP.objVal - planning_cost
        # obj
        file.write("Objective: {};\n".format(obj))
        # gap
        file.write("Gap: {};\n".format(self.MP.MIPGap))
        file.write("==============================\n")
        # time
        file.write(
            "Solving time: {} seconds;\n".format(runtime)
        )
        file.write("Iterations: {};\n".format(iteration))
        file.write(
            "Dual time: {} seconds;\n".format(runtime_dual)
        )
        # first stage results
        file.write("==============================\n")
        # planning cost
        file.write("Planning cost: {}\n".format(planning_cost))
        file.write("    Resource cost: {}\n".format(np.sum([
            self.G.nodes[i]['c_r'] * self.var_x[i].X
            for i in self.G.nodes.keys()
        ])))
        file.write("    Service cost: {}\n".format(np.sum([
            self.G.edges[i, j]['c_s'] * self.var_u[i, j].X
            for i in self.V_D
            for j in self.V_I if j in self.G.neighbors(i)
        ])))
        file.write("------------------------------\n")
        file.write("Resource allocation:\n")
        for i in self.G.nodes.keys():
            file.write("    At {}: {}\n".format(i, self.var_x[i].X))
        file.write("Service:\n")
        for i in self.V_D:
            for j in self.V_I:
                if j in self.G.neighbors(i):
                    if self.var_u[i, j].X > 0.9:
                        file.write("    ({}, {}), cost: {}\n".format(
                            i, j, self.G.edges[i, j]['c_s']
                        ))
        # MDP values
        file.write("==============================\n")
        file.write("CMDP obj: {}\n".format(self.DP.ObjVal))
        file.write("------------------------------\n")
        for s in self.S:
            file.write("    State: {}, value: {}\n".format(
                self.S_lst[s], self.var_v[s].X
            ))
        file.write("------------------------------\n")
        # policy
        policy = {}
        for s in self.S:
            opt_a = self.A[s][np.argmax([
                self.CMDP_constr[s, a].Pi
                for a in self.A[s]
            ])]
            policy[s] = opt_a
        # write policy
        for s in self.S:
            file.write("    State: {}, policy: {}\n".format(
                self.S_lst[s], self.A_lst[policy[s]]
            ))
        # pickle.dump(policy, open(
        #     'results/solutions/{}_policy.pickle'.format(name), 'wb'
        # ))
        file.write("------------------------------\n")
        # constraints
        file.write("Resource usage: \n")
        for i in self.G.nodes.keys():
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
            'obj': self.MP.ObjVal, 'runtime': runtime, 'gap': self.MP.MIPGap,
            'policy': policy, 'R': {
                (s, a): self.var_r[s, a].X
                for s in self.S for a in self.A[s]
            }, 'x': {
                i: self.var_x[i].X
                for i in self.G.nodes.keys()
            }, 'u': {
                (i, j): self.var_u[i, j].X
                for i in self.V_D for j in self.V_I
            }
        }

    # Bedners - MP
    def __build_MP(self):
        """Build MP for Bedners"""
        # model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("DualReductions", 1)
        model.setParam("IntFeasTol", 1e-5)
        model.setParam("MIPFocus", 3)
        model.setParam("NumericFocus", 0)
        model.setParam("Presolve", -1)
        # variables
        for i in self.G.nodes.keys():
            # x, amount of resource at n
            self.var_x[i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.INTEGER,
                name="x_{}".format(i)
            )
        # neighbor of independent nodes
        for i in self.V_D:
            for j in self.V_I:
                if j in self.G.neighbors(i):
                    # u, CIS service
                    self.var_u[i, j] = model.addVar(
                        lb=0, ub=1,
                        vtype=grb.GRB.BINARY,
                        name="u_{}_{}".format(i, j)
                    )
        # (s, a) pair
        for s in self.S:
            for a in self.A[s]:
                # r, reward
                self.var_r[s, a] = model.addVar(
                    lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name="r_{}_{}".format(s, a)
                )
            for i in self.V_D:
                self.var_sigma[s, i] = model.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name="sigma_{}_{}".format(s, i)
                )
        # theta, MDP value
        self.var_theta = model.addVar(
            lb=-grb.GRB.INFINITY, ub=1e5,
            vtype=grb.GRB.CONTINUOUS,
            name="theta"
        )
        model.update()
        # obj
        obj = grb.quicksum([
            # resource cost
            -1 * grb.quicksum([
                self.G.nodes[i]['c_r'] * self.var_x[i]
                for i in self.G.nodes.keys()
            ]),
            # service cost
            -1 * grb.quicksum([
                self.G.edges[i, j]['c_s'] * self.var_u[i, j]
                for i in self.V_D
                for j in self.V_I if j in self.G.neighbors(i)
            ]),
            # MDP value
            self.var_theta
        ])
        # set obj
        model.setObjective(obj, grb.GRB.MAXIMIZE)
        # constraints
        # dependent must be served by 1 independent
        for i in self.V_D:
            model.addLConstr(
                lhs=grb.quicksum([
                    self.var_u[i, j]
                    for j in self.V_I if j in self.G.neighbors(i)
                ]),
                sense=grb.GRB.GREATER_EQUAL,
                rhs=1
            )
        # budget
        model.addLConstr(
            lhs=grb.quicksum([
                # resource cost
                grb.quicksum([
                    self.G.nodes[i]['c_r'] * self.var_x[i]
                    for i in self.G.nodes.keys()
                ]),
                # service cost
                grb.quicksum([
                    self.G.edges[i, j]['c_s'] * self.var_u[i, j]
                    for i in self.V_D
                    for j in self.V_I if j in self.G.neighbors(i)
                ]),
            ]),
            sense=grb.GRB.LESS_EQUAL,
            rhs=self.B
        )
        # reward
        for s in self.S:
            for a in self.A[s]:
                model.addLConstr(
                    lhs=self.var_r[s, a],
                    sense=grb.GRB.EQUAL,
                    rhs=grb.quicksum([
                        # independent output
                        grb.quicksum([
                            self.S_lst[s][i] * self.G.nodes[i]['r']
                            for i in self.V_I
                        ]),
                        # dependent output
                        grb.quicksum([
                            (1 + self.G.nodes[i]['delta'] * (
                                self.var_sigma[s, i] - 1
                            )) * self.S_lst[s][i] * self.G.nodes[i]['r']
                            for i in self.V_D
                        ])
                    ])
                )
            # sigma
            for i in self.V_D:
                model.addLConstr(
                    lhs=self.M * self.var_sigma[s, i],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        self.S_lst[s][j] * self.var_u[i, j]
                        for j in self.V_I if j in self.G.neighbors(i)
                    ])
                )
                model.addLConstr(
                    lhs=self.var_sigma[s, i],
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=grb.quicksum([
                        self.S_lst[s][j] * self.var_u[i, j]
                        for j in self.V_I if j in self.G.neighbors(i)
                    ])
                )
        model.update()
        return model

    # Benders - DP
    def __build_DP(self):
        """Build DP as CMDP"""
        # model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("IntFeasTol", 1e-5)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 0)
        model.setParam("DualReductions", 0)
        # placeholder for var_x and var_r
        M = 100
        # value, v
        for s in self.S:
            self.var_v[s] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="v_{}".format(s)
            )
        model.update()
        # lambda, additional constraints
        for i in self.G.nodes.keys():
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
                M * self.var_lambda[i]
                for i in self.G.nodes.keys()
            ])
        ])
        model.setObjective(objective, grb.GRB.MINIMIZE)
        # MDP constraint
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
                        for i in self.G.nodes.keys()
                    ])
                ])
                self.CMDP_constr[s, a] = model.addLConstr(
                    lhs=self.CMDP_lhs[s, a],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=M
                )
        model.update()
        return model

    # Benders - update DP
    def __update_DP(self):
        """Update DP"""
        # objective
        for i in self.G.nodes.keys():
            self.var_lambda[i].Obj = self.var_x[i].X
        # rhs
        for s in self.S:
            for a in self.A[s]:
                self.CMDP_constr[s, a].rhs = self.var_r[s, a].X
        # self.DP.reset(0)
        # return
        return

    # NDPI in one algorithm
    def NDPI(self, epsilon=0.001, file_dir='None'):
        """
        Benders decomposition, two stages.
        MP: planning phase;
        SP: CMDP
        """
        logging.info("Solving NDPI...")
        runtime = time.time()
        runtime_dual = 0
        # MP
        MP = grb.Model()
        MP.setParam("OutputFlag", False)
        MP.setParam("DualReductions", 1)
        MP.setParam("IntFeasTol", 1e-5)
        MP.setParam("MIPFocus", 3)
        MP.setParam("NumericFocus", 0)
        MP.setParam("Presolve", -1)
        # variables
        var_x, var_u, var_r, var_sigma = {}, {}, {}, {}
        for i in self.G.nodes.keys():
            # x, amount of resource at n
            var_x[i] = MP.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.INTEGER,
                name="x_{}".format(i)
            )
        # neighbor of independent nodes
        for i in self.V_D:
            for j in self.V_I:
                if j in self.G.neighbors(i):
                    # u, CIS service
                    var_u[i, j] = MP.addVar(
                        lb=0, ub=1,
                        vtype=grb.GRB.BINARY,
                        name="u_{}_{}".format(i, j)
                    )
        # (s, a) pair
        for s in self.S:
            var_r[s] = MP.addVars(
                self.A[s], lb=-float('inf'), ub=float('inf')
            )
            for i in self.V_D:
                var_sigma[s, i] = MP.addVar(
                    lb=0, ub=1,
                    vtype=grb.GRB.BINARY,
                    name="sigma_{}_{}".format(s, i)
                )
        # theta, MDP value
        var_theta = MP.addVar(
            lb=-grb.GRB.INFINITY, ub=1e5,
            vtype=grb.GRB.CONTINUOUS,
            name="theta"
        )
        MP.update()
        # obj
        obj = grb.quicksum([
            # resource cost
            -1 * grb.quicksum([
                self.G.nodes[i]['c_r'] * var_x[i]
                for i in self.G.nodes.keys()
            ]),
            # service cost
            -1 * grb.quicksum([
                self.G.edges[i, j]['c_s'] * var_u[i, j]
                for i in self.V_D
                for j in self.V_I if j in self.G.neighbors(i)
            ]),
            # MDP value
            var_theta
        ])
        # set obj
        MP.setObjective(obj, grb.GRB.MAXIMIZE)
        # constraints
        # dependent must be served by 1 independent
        for i in self.V_D:
            MP.addLConstr(
                lhs=grb.quicksum([
                    var_u[i, j]
                    for j in self.V_I if j in self.G.neighbors(i)
                ]),
                sense=grb.GRB.GREATER_EQUAL,
                rhs=1
            )
        # budget
        MP.addLConstr(
            lhs=grb.quicksum([
                # resource cost
                grb.quicksum([
                    self.G.nodes[i]['c_r'] * var_x[i]
                    for i in self.G.nodes.keys()
                ]),
                # service cost
                grb.quicksum([
                    self.G.edges[i, j]['c_s'] * var_u[i, j]
                    for i in self.V_D
                    for j in self.V_I if j in self.G.neighbors(i)
                ]),
            ]),
            sense=grb.GRB.LESS_EQUAL,
            rhs=self.B
        )
        # reward
        for s in self.S:
            for a in self.A[s]:
                MP.addLConstr(
                    # lhs=var_r[s, a],
                    lhs=var_r[s].select(a)[0],
                    sense=grb.GRB.EQUAL,
                    rhs=grb.quicksum([
                        # independent output
                        grb.quicksum([
                            self.S_lst[s][i] * self.G.nodes[i]['r']
                            for i in self.V_I
                        ]),
                        # dependent output
                        grb.quicksum([
                            (1 + self.G.nodes[i]['delta'] * (
                                var_sigma[s, i] - 1
                            )) * self.S_lst[s][i] * self.G.nodes[i]['r']
                            for i in self.V_D
                        ])
                    ])
                )
            # sigma
            for i in self.V_D:
                MP.addLConstr(
                    lhs=self.M * var_sigma[s, i],
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=grb.quicksum([
                        self.S_lst[s][j] * var_u[i, j]
                        for j in self.V_I if j in self.G.neighbors(i)
                    ])
                )
                MP.addLConstr(
                    lhs=var_sigma[s, i],
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=grb.quicksum([
                        self.S_lst[s][j] * var_u[i, j]
                        for j in self.V_I if j in self.G.neighbors(i)
                    ])
                )
        MP.update()
        # mp
        mp = grb.Model()
        mp.setParam("OutputFlag", False)
        var_nu, var_psi, var_rho, dp_obj = {}, {}, {}, {}
        # nu
        for s in self.S:
            var_nu[s] = mp.addVar(
                lb=-self.M, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="nu_{}".format(s)
            )
        # rho
        for i in self.G.nodes.keys():
            var_rho[i] = mp.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="rho_{}".format(i)
            )
        objective = grb.quicksum([
            grb.quicksum([
                self.alpha[s] * var_nu[s]
                for s in self.S
            ]),
            grb.quicksum([
                var_rho[i] * self.M
                for i in self.G.nodes.keys()
            ])
        ])
        mp.setObjective(objective, grb.GRB.MINIMIZE)
        # obj for dp
        for s in self.S:
            var_psi[s] = mp.addVars(
                self.A[s], lb=-float('inf'), ub=float('inf')
            )
            for a in self.A[s]:
                dp_obj[s, a] = grb.quicksum([
                    self.gamma * grb.quicksum([
                        self.T[s, a][s_n] * var_nu[s_n]
                        for s_n in self.T[s, a].keys()
                    ]),
                    -1 * grb.quicksum([
                        var_rho[i] * self.A_lst[a][i]
                        for i in self.G.nodes.keys()
                    ])
                ])
                mp.addLConstr(
                    # lhs=var_psi[s, a],
                    lhs=var_psi[s].select(a)[0],
                    sense=grb.GRB.EQUAL,
                    rhs=dp_obj[s, a]
                )
        mp.update()
        # start iteration
        iteration, optimal = 0, False
        policy_time = 0
        mp_cuts = {}
        while not optimal:
            # time limit
            if time.time() - runtime > self.time_limit:
                break
            # solve MP
            MP.optimize()
            logging.info(f"{iteration} - {MP.ObjVal}")
            # print(MP.ObjVal)
            # modify DP and solve
            dual_start = time.time()
            # start PDPI iteration
            k, optimal_PDPI = 0, False
            # update mp
            for i in self.G.nodes.keys():
                var_rho[i].Obj = var_x[i].X
            for cut in mp_cuts.values():
                mp.remove(cut)
            pi, mp_cuts = {}, {}
            # inner iteration
            while not optimal_PDPI:
                # solve MP
                mp.optimize()
                # time limit
                if time.time() - runtime > self.time_limit:
                    break
                # control parameters
                optimal_PDPI = True
                # loop states
                for s in self.S:
                    policy_time_start = time.time()
                    # update dp
                    q = np.sum([
                        MP.getAttr("X", var_r[s]).select(),
                        mp.getAttr("X", var_psi[s]).select()
                    ], axis=0)
                    q_ind = np.argmax(q)
                    pi[s] = self.A[s][q_ind]
                    policy_time += time.time() - policy_time_start
                    # comapre state value
                    if np.abs(var_nu[s].X - q[q_ind]) > epsilon or k < 1:
                        optimal_PDPI = False
                        # PDPI cut
                        if (s, pi[s]) in mp_cuts.keys():
                            continue
                        mp_cuts[s, pi[s]] = mp.addLConstr(
                            lhs=var_nu[s],
                            sense=grb.GRB.GREATER_EQUAL,
                            rhs=grb.quicksum([
                                var_r[s].select(pi[s])[0].X,
                                var_psi[s].select(pi[s])[0]
                            ])
                        )
                k += 1
            # print(time.time() - dual_start)
            runtime_dual += time.time() - dual_start
            # compare MDP value
            if np.abs(var_theta.X - mp.ObjVal) > epsilon:
                optimal = False
                val_y = {}
                for key in mp_cuts.keys():
                    if mp_cuts[key].Pi != 0:
                        val_y[key] = mp_cuts[key].Pi
                # GBD cut
                MP.addLConstr(
                    lhs=var_theta,
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=grb.quicksum([
                        grb.quicksum([
                            val_y[key] * grb.quicksum([
                                var_r[key[0]].select(key[1])[0],
                                -1 * var_nu[key[0]].X,
                                var_psi[key[0]].select(key[1])[0].X
                            ])
                            for key in val_y.keys()
                        ]),
                        grb.quicksum([
                            self.alpha[s] * var_nu[s].X
                            for s in self.S
                        ]),
                        grb.quicksum([
                            var_rho[i].X * var_x[i]
                            for i in self.G.nodes.keys()
                        ])
                    ])
                )
            else:
                optimal = True
            iteration += 1
        # policy
        self.policy = pi
        # time
        runtime = time.time() - runtime
        # print solution to file
        if file_dir == "None":
            return
        file = open("{}/{}-NDPI.txt".format(file_dir, self.name), "w+")
        file.write("==============================\n")
        # not optimal
        if MP.status != grb.GRB.OPTIMAL:
            file.write("Optimal status {};\n".format(MP.status))
            return {
                'obj': float('nan'), 'runtime': runtime,
                'gap': float('nan')
            }
        # calculate obj
        planning_cost = -(MP.ObjVal - var_theta.X)
        obj = mp.objVal - planning_cost
        # obj
        file.write("Objective: {};\n".format(obj))
        # gap
        file.write("Gap: {};\n".format(MP.MIPGap))
        file.write("==============================\n")
        # time
        file.write(
            "Solving time: {} seconds;\n".format(runtime)
        )
        file.write("Iterations: {};\n".format(iteration))
        file.write(
            "Dual time: {} seconds;\n".format(runtime_dual)
        )
        # first stage results
        file.write("==============================\n")
        # planning cost
        file.write("Planning cost: {}/{}\n".format(planning_cost, self.B))
        file.write("    Resource cost: {}\n".format(np.sum([
            self.G.nodes[i]['c_r'] * var_x[i].X
            for i in self.G.nodes.keys()
        ])))
        file.write("    Service cost: {}\n".format(np.sum([
            self.G.edges[i, j]['c_s'] * var_u[i, j].X
            for i in self.V_D
            for j in self.V_I if j in self.G.neighbors(i)
        ])))
        file.write("------------------------------\n")
        file.write("Resource allocation:\n")
        for i in self.G.nodes.keys():
            file.write("    At {}: {}\n".format(i, var_x[i].X))
        file.write("Service:\n")
        for i in self.V_D:
            for j in self.V_I:
                if j in self.G.neighbors(i):
                    if var_u[i, j].X > 0.9:
                        file.write("    ({}, {}), cost: {}\n".format(
                            i, j, self.G.edges[i, j]['c_s']
                        ))
        # MDP values
        file.write("==============================\n")
        file.write("CMDP obj: {}\n".format(mp.objVal))
        file.write("------------------------------\n")
        for s in self.S:
            file.write("    State: {}, value: {}\n".format(
                self.S_lst[s], var_nu[s].X
            ))
        file.write("------------------------------\n")
        # write policy
        for s in self.S:
            file.write("    State: {}, policy: {}\n".format(
                self.S_lst[s], self.A_lst[pi[s]]
            ))
        # pickle.dump(policy, open(
        #     'results/solutions/{}_policy.pickle'.format(name), 'wb'
        # ))
        file.write("------------------------------\n")
        # constraints
        file.write("Resource usage: \n")
        for i in self.G.nodes.keys():
            file.write("    Usage at {}: {:.2f}, available: {:.2f}\n".format(
                i, np.sum([
                    self.A_lst[key[1]][i] * val_y[key]
                    for key in val_y
                ]), var_x[i].X
            ))
        file.close()
        # return
        return {
            'obj': MP.ObjVal, 'runtime': runtime, 'gap': MP.MIPGap,
            'policy': pi, 'R': {
                (s, a): var_r[s].select(a)[0].X
                for s in self.S for a in self.A[s]
            }, 'x': {
                i: var_x[i].X
                for i in self.G.nodes.keys()
            }, 'u': {
                (i, j): var_u[i, j].X
                for i in self.V_D for j in self.V_I
            }
        }

    def simulate(self, x, u, R, policy, file_dir='None'):
        """
        Run simulation
        """
        # number of attacks
        attacks = np.random.choice(range(
            int(len(self.G.nodes()) - 0.50 * len(self.G.nodes())),
            int(len(self.G.nodes()) + 0.50 * len(self.G.nodes()) + 1)
        ))
        # history
        state_hist, action_hist, reward_hist = [], [], []
        dependency_loss_hist = []
        defended_attacks_hist = []
        # initial state
        state = tuple([1] * len(self.G.nodes()))
        s = self.S_lst.index(state)
        state_hist.append(state)
        # loops
        t = 1
        while True:
            # action
            action = list(self.A_lst[policy[s]])
            # index of adjusted action
            a = self.A_lst.index(tuple(action))
            action_hist.append(action)
            # reward
            reward = R[s, a]
            reward_hist.append(reward)
            # new epoch
            t += 1
            # new state
            new_state = self.S_lst[np.random.choice(
                a=list(self.T[s, a].keys()), size=1, replace=False,
                p=list(self.T[s, a].values())
            )[0]]
            # defended attacks
            if new_state == state:
                defended_attacks_hist.append(1)
            else:
                defended_attacks_hist.append(0)
            # dependency loss
            V_D_loss = 0
            for i in self.V_D:
                if np.sum([u[i, j] * new_state[j] for j in self.V_I]) < 0.9:
                    V_D_loss += self.G.nodes[i]['delta'] * self.G.nodes[i]['r']
            dependency_loss_hist.append(V_D_loss)
            # terminate
            if t > attacks or np.sum(list(new_state)) == 0:
                break
            else:
                # continue
                state = new_state
                s = self.S_lst.index(state)
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
