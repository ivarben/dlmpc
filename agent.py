import numpy as np
import scipy.linalg as la
import cvxpy as cp
import matplotlib.pyplot as pl

class Agent:
    def __init__(self, id, A, B, Q, R, xS, xF, N, J, time):
        self.id = id
        self.step_nr = 0
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
        self.neighbours = [] # list of neighbouring agents, will include self for ordering purposes
        self.neighbour_ids = [] # store id of neighbour for later message processing
        self.next = None # defines the order of sequential optimization
        self.previous = None # same order but reversed

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
        self.x_neighbours_test = np.zeros((self.n, self.N+1, len(self.neighbours)))

    def update_x_neighbours(self):
        return

    def evaluate_stage_cost(self, x, u):
        return x@self.Q@x + np.array([0, -40])@x + 400 + 0.1*x[1]*x[1] + 0.1*x[1] + u@self.R@u

    def initiate_LSS(self): # Local Safe Set
        dim = 1 # space for ctg

        for agent in self.neighbours:
            dim += agent.n # space for own and neighbours state variables

        self.LSS = np.empty([dim, 0]) # dim rows and 0 columns (yet)
        self.LSSC = np.empty([self.m, 0]) # to store safe control associated with LSS
        self.SS = np.empty([1+self.n, 0]) # to store own safe states associated with LSS (and ctg)

    def append_LSS(self, j): # needs iteration number
        J0 = np.cumsum(self.stage_costs[:,:,j], axis = 1) # cumulative stage costs over time
        ctg = J0[:,-1] - J0 # cost to go calculation
        entry = ctg ### start with only cost

        for neighbour in self.neighbours:
                entry = np.concatenate((entry, neighbour.x_trajectory[:,:,j]), axis = 0)

        self.LSS = np.concatenate((self.LSS, entry), axis = 1)
        self.LSSC = np.concatenate((self.LSSC, self.u_trajectory[:,:,j]), axis = 1)
        self.SS = np.concatenate((self.SS, np.concatenate((ctg, self.x_trajectory[:,:,j]), axis = 0)), axis = 1)

    def update(self, x, u):
        return self.A@x + self.B@u

    def sort(self, id): # used for determining the index for info in incoming messages
        index = 0
        for n_id in self.neighbour_ids:
            if id > n_id:
                index += 1
        return index

    def calculate_J_F(self, x_neighbours, x_pred, u_pred):
        # F_old found from matching planned terminal state with ctg from LSS
        for i in range(len(self.neighbours)):
            if i == 0:
                planned = x_neighbours[:,self.N,i]
            else:
                planned = np.concatenate((planned, x_neighbours[:,self.N,i]), axis = 0)

        ctgs = [] # list of ctgs in case there are several matching elements in LSS
        for i in range(self.LSS.shape[1]):
            if np.array_equal(planned, self.LSS[1:,i]):
                ctgs.append(self.LSS[0,i])
        F = min(ctgs)

        stage_costs = np.zeros(self.N-1)
        for i in range(stage_costs.shape[0]):
            stage_costs[i] = self.evaluate_stage_cost(x_pred[:,i], u_pred[:,i]) # evaluate stage costs for hat variables
        J = np.sum(stage_costs) + F # evaluate total costs for hat variables

        return J, F

    def proceed(self, k):
        self.step_nr += 1

        if self.previous is None:
            if self.step_nr == 1:
                if k == 0:
                    self.step2(k)
                else:
                    self.step1(k)
            elif self.step_nr == 2:
                self.step2(k)
            elif self.step_nr == 3:
                self.step3(k)
            elif self.step_nr == 4:
                self.step4(k)
        else:
            self.previous.proceed(k)

    def step0(self):
        self.x_hat = self.SS[1:,:self.N + 1] # take initial x_hat from safe set
        self.u_hat = self.LSSC[:,:self.N]

        for neighbour in self.neighbours:
            neighbour.x_neighbours[:,:,neighbour.sort(self.id)] = self.x_hat

        if self.next is None:
            self.proceed(0)
        else:
            self.next.step0()

    def step1(self, k):
        # update hat variables and send, similar to step 0
        return

    def step2(self, k): # compute J_old and F_old

        self.J_old, self.F_old = self.calculate_J_F(self.x_neighbours, self.x_hat, self.u_hat)

        if self.next is not None:
            self.next.step2(k)
        else:
            self.proceed(k)

    def step3(self, k):
        ### A ### Solve MPC problem
        self.x_test, self.u_test, self.J_test, self.F_test = self.solve_MPC()
        ### B ### Let neigbours calculate resulting costs
        for neighbour in self.neighbours:
            neighbour.x_neighbours_test = neighbour.x_neighbours # x_neighbours_test should be the same as x_neighbours...
            neighbour.x_neighbours_test[:,:,neighbour.sort(self.id)] = self.x_test # ... but with updated information from the current agent

            if neighbour.id < self.id:
                neighbour.x_bar = neighbour.x_star
                neighbour.u_bar = neighbour.u_star
            else:
                neighbour.x_bar = neighbour.x_hat
                neighbour.u_bar = neighbour.u_hat

            # calculate F and J
            neighbour.J_test, neighbour.F_test = neighbour.calculate_J_F(neighbour.x_neighbours_test, neighbour.x_bar, neighbour.u_bar)

        ### C ### Neighbours give feedback
        ### B and C can be combined in one loop for efficiency

        d = self.J_test - self.J_old # own total cost change
        e = self.F_test - self.F_old # own terminal cost change

        for neighbour in self.neighbours:
            d += (neighbour.J_test - neighbour.J_old)
            e += (neighbour.F_test - neighbour.F_old)

        check = (d <= 0 and e <= 0) ## Bool to check conditions if a decrease in both total and termnal cost of the overall system was reached
        if check: # Use optimized variables
            self.x_star = self.x_test
            self.u_star = self.u_test
            flag = 1
        else: # use old, hat variables
            self.x_star = self.x_hat
            self.u_star = self.u_hat
            flag = 0
        ### D ### Neighbours update info from self
        for neighbour in self.neighbours:
            if flag == 1:
                neighbour.x_neighbours = neighbour.x_neighbours_test
                neighbour.J_old = neighbour.J_test
                neighbour.F_old = neighbour.F_test

        if self.next is not None:
            self.next.step3(k)
        else:
            self.proceed(k)

    def step4(self, k):
        return

    def solve_MPC(self):
        ### Idea: Subset LSS to those states that are compatible with neighbour states
        x_neighbours_stacked = self.x_neighbours[:,:,0]
        for i in range(1, self.x_neighbours.shape[2]): ### Not super neat, fix with np.reshape() if possible
            x_neighbours_stacked = np.concatenate((x_neighbours_stacked, self.x_neighbours[:,:,i]), axis = 0)

        SS_feasible = np.zeros((self.SS.shape[0], 0)) # denotes safe terminal states

        for i in range(self.LSS.shape[1]):
            if np.array_equal(self.LSS[1:,i], x_neighbours_stacked[:,self.N]): # compare states (not including ctg)
                SS_feasible = np.concatenate((SS_feasible, self.SS[:,[i]]), axis = 1) # append state from SS (including ctg)

        SS = cp.Parameter((SS_feasible.shape[0], SS_feasible.shape[1]), value = SS_feasible)
        x_neighbours = cp.Parameter((x_neighbours_stacked.shape[0], x_neighbours_stacked.shape[1]), value = x_neighbours_stacked)
        x = cp.Variable((self.n, self.N+1)) # state variables
        u = cp.Variable((self.m, self.N)) # control variables
        delta = cp.Variable(SS_feasible.shape[1], boolean = True) ## Binary variable to model inclusion in discrete set

        stage_costs = 0
        constr = []
        for k in range(self.N):
            constr += [x[:,k+1] == self.A@x[:,k] + self.B@u[:,k]] # Dynamics
            stage_costs += cp.quad_form(x[:,k], self.Q) + np.array([0, -40])@x[:,k] + 400 + 0.1*x[1,k] + cp.quad_form(u[:,k], self.R) # stage costs

        for k in range(self.N):
            for i in range(len(self.neighbours)):
                if i > self.sort(self.id):
                    constr += [x[0,k] - x_neighbours[i*self.n,k] >= 20]
                elif i < self.sort(self.id):
                    constr += [x_neighbours[i*self.n,k] - x[0,k] >= 20]

        terminal_cost = cp.sum(cp.multiply(delta, np.squeeze(SS.value[0,:]))) # terminal cost
        constr += [cp.sum(delta) == 1] # only one terminal state
        constr += [x[:,self.N] == SS[1:,:]@delta] # terminal state has to be in reduced LSS
        constr += [x[:,0] == np.squeeze(self.x_hat[:,0])] # initial condition

        problem = cp.Problem(cp.Minimize(stage_costs + terminal_cost), constr)
        problem.solve(solver=cp.GUROBI)

        return x.value, u.value, problem.value, terminal_cost.value
