from dlmpc import *

def main():
    T = 100
    G = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) # 3 node path-graph
    A = np.matrix("1, 1; 0, 1") # Double integrator
    B = np.matrix("0; 1")       # system dynamics

    N = 3 # control/ prediction horizon
    J = 1 # Number of iterations of LMPC

    dlmpc = DLMPC(T, G, A, B, N, J) # DLMPC instance

    dlmpc.solve() # Solve the DLMPC

    dlmpc.plot() # plot the results

if __name__ == '__main__':
    main()
