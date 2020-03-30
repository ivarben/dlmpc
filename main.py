from dlmpc import *

def main():
    T = 100
    G = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]) # 3 node path-graph
    A = np.matrix("1, 1; 0, 1") # Double integrator
    B = np.matrix("0; 1")       # system dynamics

    N = 5 # control/ prediction horizon
    J = 2 # Number of iterations of LMPC

    dlmpc = DLMPC(T, G, A, B, N, J) # DLMPC instance

    dlmpc.solve() # Solve the DLMPC

    dlmpc.plot() # plot the results

if __name__ == '__main__':
    main()
