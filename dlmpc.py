from agent import *

class DLMPC:
    def __init__(self, T, G, A, B, N, J,):
        self.T = T # Optimize for T time steps
        self.time = np.arange(T + 1)
        self.G = G # Graph adjacency matrix
        self.V = self.G.shape[0] # number of agents
        self.A = A # A matrix for system dynamics
        self.B = B # B matrix for control dynamics
        self.N = N # Control/ prediction horizon of each MPC problem
        self.J = J # Number of iterations of LMPC

        self.agents = [] # List of agents
        self.xS = [] # Initial state of the respective agents
        self.xF = [] # Goal state of the respective agents

        for i in range(self.V): # hardcoding to decide some initial and final states
            self.xS.append(np.array([50*(self.V - i - 1), 0]))
            self.xF.append(np.array([self.xS[i] + 1000, 0]))

        Q = np.array([[0, 0], [0, 1]])
        R = np.array([[1]])

        for i in range(self.V): # append agent objects to agent list
            self.agents.append(Agent(i+1, self.A, self.B, Q, R, self.xS[i], self.xF[i], self.N, self.J, self.time)) # each with same dynamics but different tasks

        for i in range(self.V):
            for j in range(self.V):
                if self.G[i, j] == 1: # 1 means that there's an edge
                    self.agents[i].add_neighbour(self.agents[j]) # append neighbours according to adjacencies in G

        for i in range(self.V):
            if i != self.V - 1:
                self.agents[i].next = self.agents[i+1] # link agents to allow knowledge of who comes next
            if i != 0:
                self.agents[i].previous = self.agents[i-1] # link in reverse direction for back-communication
            self.agents[i].initiate_x_neighbours()
            self.agents[i].initiate_LSS() # initiate local safe sets

    def iteration_0(self):
        for agent in self.agents:
            agent.x_trajectory[:,0,0] = agent.xS
            agent.u_trajectory[:,0,0] = 10 # hardcoded acceleration
            agent.stage_costs[:,0,0] = agent.evaluate_stage_cost(agent.x_trajectory[:,0,0], agent.u_trajectory[:,0,0])

        for t in self.time[1:]: # iterate time
            for agent in self.agents: # same for each agent
                agent.x_trajectory[:,t,0] = agent.update(agent.x_trajectory[:,t-1,0], agent.u_trajectory[:,t-1,0]) # Dynamics

                if t != self.time.shape[0] - 1:
                    agent.u_trajectory[:,t,0] = 0 # hardcoded acceleration
                else:
                    agent.u_trajectory[:,t,0] = -10 # hardcoded acceleration

                agent.stage_costs[0,t,0] = agent.evaluate_stage_cost(agent.x_trajectory[:,t,0], agent.u_trajectory[:,t,0])

        for agent in self.agents:
            agent.append_LSS(0)

        ### Note:: Should DLMPC be able to access agent variables?
        ## For SS0 it doesn't really matter, don't have to scrap this code.
        ## But for the rest of the algorithm, it should be compoletely decentralized. I.e. agent 1 starts doing its stuff and agent 2
        ## doesn't start until it has received the information from one, and so on.
        ## ROS? Event handling? Observer pattern?

    def iterate(self):
        # Step 0
        self.agents[0].step0()

        for t in self.time[1:]:
            pass
            # Step 3
            #self.agents[0].optimize() #
            # call comunicate() (basically the DMPC algorithm from the paper)

    def solve(self):
        self.iteration_0()
        for j in range(self.J): # Iteration number j+1
            self.iterate()
        return

    def plot(self):
        pl.figure()

        for agent in self.agents:
            pl.plot(self.time, agent.x_trajectory[0,:,0])

        pl.show()
