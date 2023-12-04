import numpy as np
import matplotlib.pyplot as plt

from dpilqr import split_agents, plot_solve
import dpilqr
from dpilqr.cost import GameCost, GameCost2, GammaCost, ProximityCost, ObstacleCost, ReferenceCost, ConnectivityMaintenanceGameCost, ConnectivityMaintenanceGameCost2
from dpilqr.dynamics import (
    DoubleIntDynamics4D,
    UnicycleDynamics4D,
    QuadcopterDynamics6D,
    MultiDynamicalModel,
)
from dpilqr.distributed import solve_rhc
from dpilqr.problem import ilqrProblem
from dpilqr.util import split_agents_gen, random_setup
import scenarios

π = np.pi
g = 9.80665

def los_di():
    n_states = 4
    n_controls = 2
    n_agents = 3
    x_dims = [n_states] * n_agents
    u_dims = [n_controls] * n_agents
    n_dims = [2] * n_agents
    n_d = n_dims[0]

    x0 = np.array([[0.5, 0, 0, 0,
                    2.5, 0, 0, 0,
                    1.5, 0, 0, 0]]).T
    xf = np.array([[0.5, 4, 0, 0,
                    2.5, 4, 0, 0,
                    1.5, 4, 0, 0]]).T
    obstacles = [[2.0, 2.0], [1.0, 2.0]]
    dpilqr.eyeball_scenario(x0, xf, n_agents, n_states, obstacles=obstacles)
    plt.show()

    dt = 0.05
    N = 60

    tol = 1e-6
    ids = [100 + i for i in range(n_agents)]

    model =dpilqr.DoubleIntDynamics4D
    dynamics = dpilqr.MultiDynamicalModel([model(dt, id_) for id_ in ids])

    Q = 1.0 * np.diag([1, 1] + [0] * (n_states - 2))
    R = np.eye(2)
    Qf = 1000.0 * np.eye(Q.shape[0])
    radius = 0.5

    goal_costs = [
        ReferenceCost(xf_i, Q.copy(), R.copy(), Qf.copy(), id_)
        for xf_i, id_ in zip(split_agents_gen(xf, x_dims), ids)
    ]
    prox_cost = dpilqr.ProximityCost(x_dims, radius, n_dims)

    obst_cost = ObstacleCost(x_dims, radius, n_dims, obstacles)
    # game_cost = dpilqr.GameCost(goal_costs, prox_cost, obst_cost=obst_cost)
    # game_cost = ConnectivityMaintenanceGameCost(x_dims, n_dims, goal_costs, obstacles, prox_cost, obst_cost)
    # game_cost = ConnectivityMaintenanceGameCost2(x_dims, n_dims, goal_costs, prox_cost, obstacles)

    gamma_cost = GammaCost(x_dims, n_dims, obstacles)
    game_cost = GameCost2(goal_costs, prox_cost, obst_cost=obst_cost, gamma_cost=gamma_cost)

    problem = ilqrProblem(dynamics, game_cost)

    # Solve the problem centralized.
    print("\t\t\tcentralized")
    Xc, Uc, Jc = solve_rhc(
        problem,
        x0,
        N,
        radius,
        centralized=True,
        n_d=n_d,
        step_size=3,
        t_kill=None,
        t_diverge=4*N*dt,
        dist_converge=0.2,
        use_L=True
    )

    plt.clf()
    plot_solve(Xc, Jc, xf.T, x_dims, True, n_d, obstacles=obstacles)

    plt.figure()
    dpilqr.plot_pairwise_distances(Xc, x_dims, n_dims, radius)

    dpilqr.graphics.plot_pairwise_gamma(Xc, x_dims, n_dims, obstacles)
    plt.show()

    dpilqr.make_trajectory_gif(f"{n_agents}-los.gif", Xc, xf, x_dims, radius, obstacles)

def main():
    los_di()

if __name__ == "__main__":
    main()