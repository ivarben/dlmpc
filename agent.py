import numpy as np
import scipy.linalg as la
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as pl

class Agent:
    def __init__(self, id, A, B, Q, R, xS, xF, N, J, time):
        self.id = id
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.xS = xS # initial state
        self.xF = xF # final state
        self.N = N
        self.time = time
        self.neighbours = [] # list of neighbouring agents
        self.neighbour_ids = [] # store id of neighbour for later message processing
        self.next = None # defines the order of sequential optimization

        # data which needs to be stored:
        self.stage_costs = np.zeros((1, time.shape[0], J+1)) # realized stage costs over time, saves all iterations. Used for ctg.

        self.x_trajectory = np.zeros((self.n, time.shape[0], J+1)) # realized state trajectories (saves all iterations, even 0)
        self.u_trajectory = np.zeros((self.m, time.shape[0], J+1)) # realized control trajectories (saves all iterations, even 0)

        self.x_hat = np.zeros((self.n, self.N + 1)) # candidate state prediction
        self.u_hat = np.zeros((self.m, self.N)) # candidate control prediction

        self.x_star = np.zeros((self.n, self.N + 1, time.shape[0])) # optimal state prediction (saves all time steps)
        self.u_star = np.zeros((self.m, self.N, time.shape[0])) # optimal control prediction (saves all time steps)

        self.x_test = np.zeros((self.n, self.N + 1)) # state solution to MPC problem
        self.u_test = np.zeros((self.m, self.N)) # control solution to MPC problem

        self.x_bar = np.zeros((self.n, self.N + 1)) # state prediction used to evaluate neighbours' solutions
        self.u_bar = np.zeros((self.m, self.N)) # control prediction used to evaluate neighbours' solutions

        self.J_old = 0 # total cost for hat variables
        self.F_old = 0 # terminal cost for hat variables

        self.J_test = 0 # new total cost with the test varaibles (for when both active and not)
        self.F_test = 0 # new terminal cost with the test varaibles (for when both active and not)

    def add_neighbour(self, agent):
        self.neighbours.append(agent)
        self.neighbour_ids.append(agent.id)

    def initiate_x_neighbours(self):
        self.x_neighbours = np.zeros((self.n, self.N+1, len(self.neighbours)))

    def update_x_neighbours(self):
        return

    def evaluate_stage_cost(self, x, u):
        return x@self.Q@x + u@self.R@u

    def initiate_LSS(self): # Local Safe Set
        dim = 1 + self.n # space for ctg and own state variables

        for agent in self.neighbours:
            dim += agent.n # space for neighbours state variables

        self.LSS = np.empty([dim, 0]) # dim rows and 0 columns (yet)
        self.LSSC = np.empty([self.m, 0]) # to store safe control associated with safe set

    def append_LSS(self, j): # needs iteration number
        J0 = np.cumsum(self.stage_costs[:,:,j], axis = 1) # cumulative stage costs over time
        ctg = J0[-1] - J0 # cost to go calculation
        entry = np.concatenate((ctg, self.x_trajectory[:,:,j]))

        #####
        ### TODO: CHANGE SO THAT YOUR OWN TRAJECTORY IS NOT NECESSARILY FIRST
        #####

        for neighbour in self.neighbours:
            entry = np.concatenate((entry, neighbour.x_trajectory[:,:,j]), axis = 0)

        self.LSS = np.concatenate((self.LSS, entry), axis = 1)
        self.LSSC = np.concatenate((self.LSSC, self.u_trajectory[:,:,j]), axis = 1)

    def update(self, x, u):
        return self.A@x + self.B@u

    def sort(self, id): # used for determining the index for info in incoming messages
        index = 0
        for i in self.neighbour_ids:
            if id > i:
                index += 1
        return index

    def step0(self):
        self.x_hat = self.LSS[1 + self.sort(self.id):1 + self.sort(self.id) + self.n,:self.N + 1]
        self.u_hat = self.LSSC[:,:self.N]

        for neighbour in self.neighbours:
            neighbour.x_neighbours[:,:,neighbour.sort(self.id)] = self.x_hat

        if self.next is not None:
            self.next.step0()

    def solve_MPC(self):
        return