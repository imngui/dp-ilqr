#!/usr/bin/env python

"""Implements various cost structures in the LQ Game"""

import abc
import itertools

import numpy as np
from scipy.optimize import approx_fprime

from .util import (
    Point,
    compute_pairwise_distance,
    compute_pairwise_distance_nd,
    split_agents_gen,
    uniform_block_diag,
)


class Cost(abc.ABC):
    """
    Abstract base class for cost objects.
    """

    @abc.abstractmethod
    def __call__(self, *args):
        """Returns the cost evaluated at the given state and control"""
        pass

    @abc.abstractmethod
    def quadraticize():
        """Compute the jacobians and hessians of the operating point wrt. the states
        and controls
        """
        pass


class ReferenceCost(Cost):
    """
    The cost of a state and control from some reference trajectory.
    """

    _id = 0

    def __init__(self, xf, Q, R, Qf=None, id=None):

        if Qf is None:
            Qf = np.eye(Q.shape[0])

        if not id:
            id = ReferenceCost._id
            ReferenceCost._id += 1

        # Define states as rows so that xf doesn't broadcast x in __call__.
        # self.xf = xf.reshape(1, -1)
        self.xf = xf.flatten()

        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.id = id

        self.Q_plus_QT = Q + Q.T
        self.R_plus_RT = R + R.T
        self.nx = Q.shape[0]
        self.nu = R.shape[0]

    @property
    def x_dim(self):
        return self.Q.shape[0]

    @property
    def u_dim(self):
        return self.R.shape[0]

    @classmethod
    def _reset_ids(cls):
        cls._id = 0

    def __call__(self, x, u, terminal=False):
        if not terminal:
            u = u.reshape(1, -1)
            return (x - self.xf) @ self.Q @ (x - self.xf).T + u @ self.R @ u.T
        return (x - self.xf) @ self.Qf @ (x - self.xf).T

    def quadraticize(self, x, u, terminal=False):
        x = x.flatten()
        u = u.flatten()

        L_x = (x - self.xf).T @ self.Q_plus_QT
        L_u = u.T @ self.R_plus_RT
        L_xx = self.Q_plus_QT
        L_uu = self.R_plus_RT
        L_ux = np.zeros((self.nu, self.nx))

        if terminal:
            L_x = (x - self.xf).T @ (self.Qf + self.Qf.T)
            L_xx = self.Qf + self.Qf.T
            L_u = np.zeros((self.nu))
            L_uu = np.zeros((self.nu, self.nu))

        return L_x, L_u, L_xx, L_uu, L_ux

    def __repr__(self):
        return (
            f"ReferenceCost(\n\tQ: {self.Q},\n\tR: {self.R},\n\tQf: {self.Qf}"
            f",\n\tid: {self.id}\n)"
        )


class ProximityCost(Cost):
    def __init__(self, x_dims, radius, n_dims):
        self.x_dims = x_dims
        self.radius = radius
        self.n_dims = n_dims
        self.n_agents = len(x_dims)

    def __call__(self, x):
        # print('x_dims: ', self.x_dims)
        if len(self.x_dims) == 1:
            return 0.0

        # Try to vectorize the distance computation if possible.
        if len(set(self.n_dims)) == 1:
            distances = compute_pairwise_distance(x, self.x_dims)

        # Otherwise, compute the distance to the lower of the i and j number of
        # dimensions.
        else:
            distances = compute_pairwise_distance_nd(
                x.reshape(1, -1), self.x_dims, self.n_dims
            )
        # print('x: ', x)
        # print('dist: ', distances)
        pair_costs = np.fmin(np.zeros(1), distances - self.radius) ** 2
        return pair_costs.sum()

    def quadraticize(self, x):
        nx = sum(self.x_dims)
        nx_per_agent = self.x_dims[0]

        L_x = np.zeros((nx))
        L_xx = np.zeros((nx, nx))
        for i, n_dim_i in zip(range(self.n_agents), self.n_dims):
            for j, n_dim_j in zip(range(i + 1, self.n_agents), self.n_dims[i + 1 :]):

                # Penalize distance for the minimum dimension dynamical model.
                nd = min(n_dim_i, n_dim_j)

                L_xi = np.zeros((nx))
                L_xxi = np.zeros((nx, nx))

                ix = nx_per_agent * i
                jx = nx_per_agent * j

                L_x_pair, L_xx_pair = quadraticize_distance(
                    Point(*x[..., ix : ix + nd]),
                    Point(*x[..., jx : jx + nd]),
                    self.radius,
                    nd,
                )

                L_xi[np.arange(ix, ix + nd)] = +L_x_pair
                L_xi[np.arange(jx, jx + nd)] = -L_x_pair

                L_xxi[ix : ix + nd, ix : ix + nd] = +L_xx_pair
                L_xxi[jx : jx + nd, jx : jx + nd] = +L_xx_pair
                L_xxi[ix : ix + nd, jx : jx + nd] = -L_xx_pair
                L_xxi[jx : jx + nd, ix : ix + nd] = -L_xx_pair

                L_x += L_xi
                L_xx += L_xxi

        return L_x, L_xx


class ObstacleCost(Cost):
    def __init__(self, x_dims, radius, n_dims, obstacles):
        self.x_dims = x_dims
        self.radius = radius
        self.n_dims = n_dims
        self.obstacles = obstacles
        self.n_agents = len(x_dims)

    def __call__(self, x):
        n_dim = self.n_dims[0]
        obst_distances = []
        for i in range(self.n_agents):
            for o in self.obstacles:
                dist = np.linalg.norm(x[i*n_dim:i*n_dim+2] - o)
                if dist < self.radius:
                    obst_distances.append(dist**-1)
                else:
                    obst_distances.append(0)
        return sum(obst_distances)


    def quadraticize(self, x):
        return quad_los_finite_diff(self, x)


class AlphaStarCost(Cost):
    def __init__(self, x_dims, n_dims, d_min=0.5, d_max=2.0, k_a=1.0):
        self.x_dims = x_dims
        self.n_dims = n_dims
        self.n_agents = len(x_dims)

        self.d_min = d_min
        self.d_max = d_max
        self.k_a = k_a
        self.mu_a = np.pi / (d_max - d_min)
        self.v_a = -self.mu_a * d_min

    def __call__(self, xi, xj):
        # TODO: Make sure this is correct
        # if len(self.x_dims) == 1:
        #     return 0.0
        #
        # # Try to vectorize the distance computation if possible.
        # if len(set(self.n_dims)) == 1:
        #     distances = compute_pairwise_distance(x, self.x_dims)
        #
        # # Otherwise, compute the distance to the lower of the i and j number of
        # # dimensions.
        # else:
        #     distances = compute_pairwise_distance_nd(
        #         x.reshape(1, -1), self.x_dims, self.n_dims
        #     )
        #
        # dij = distances
        dij = np.linalg.norm(xi-xj)

        if dij <= self.d_min:
            return 0.0
        elif self.d_min < dij <= self.d_max:
            return (self.k_a/2.0)*(1.0 - np.cos(self.mu_a*dij + self.v_a))
        else:
            return self.k_a

    def quadraticize(self):
        pass


class AlphaCost(Cost):
    def __init__(self, x_dims, n_dims, obstacles):
        self.x_dims = x_dims
        self.n_dims = n_dims
        self.n_agents = len(x_dims)

        self.gamma_cost = GammaCost(x_dims, n_dims, obstacles)
        self.alpha_star_cost = AlphaStarCost(x_dims, n_dims)

    def __call__(self, x, compute_L=False):
        n_states = self.x_dims[0]
        # Compute S_i
        S = [[] for i in range(self.n_agents)]
        # Compute S_j
        # Sj = [[] for j in range(self.n_agents)]
        pair_inds = np.array(list(itertools.combinations(range(self.n_agents), 2)))
        for (i, j) in pair_inds:
            S[i].append(j)
            S[j].append(i)
            # si = x[j*n_states:j*n_states+n_states]
            # sj = x[i*n_states:i*n_states+n_states]
            # S[i].append(x[j*n_states:j*n_states+n_states])
            # S[j].append(x[i*n_states:i*n_states+n_states])
        # ls = len(S)
        # si = S[0]
        a_stars = np.zeros((self.n_agents))
        idx = 0
        for (i, j) in pair_inds:
            xi = x[i*n_states:i*n_states+n_states]
            ai = 1.0
            for k in S[i]:
                # a_i *= compute_alpha_ij_star(compute_dij(x_i, x_k))
                xk = x[k*n_states:k*n_states+n_states]
                ai *= self.alpha_star_cost(xi, xk)

            aj = 1.0
            xj = x[j*n_states:j*n_states+n_states]
            for k in S[j]:
                # a_j *= compute_alpha_ij_star(compute_dij(x_j, x_k))
                if j != i:
                    xk = x[k * n_states:k * n_states + n_states]
                    aj = self.alpha_star_cost(xj, xk)
            a_stars[idx] = ai * aj
            idx += 1
        # print(a_stars)
        if compute_L:
            return a_stars.reshape((1,-1))
        return a_stars.reshape((1,-1))
        # a = sum(1.0 - a_stars.reshape((1,-1))[0])
        # return a
        # return np.array([[ai * aj, ai * aj, ai * aj]])

    def quadraticize(self, x):
        return quad_los_finite_diff(self, x)


class BetaCost(Cost):
    def __init__(self, x_dims, n_dims, k_b=1.0, omega=5.0, d0=1.0):
        self.x_dims = x_dims
        self.n_dims = n_dims
        self.n_agents = len(x_dims)

        self.k_b = k_b
        self.omega = omega
        self.d0 = d0
        pass

    def __call__(self, x):
        # TODO: Make sure this is correct
        if len(self.x_dims) == 1:
            return 0.0

        # Try to vectorize the distance computation if possible.
        # print('x: ', x)
        if len(set(self.n_dims)) == 1:
            distances = compute_pairwise_distance(x, self.x_dims)

        # Otherwise, compute the distance to the lower of the i and j number of
        # dimensions.
        else:
            distances = compute_pairwise_distance_nd(
                x.reshape(1, -1), self.x_dims, self.n_dims
            )

        dij = distances

        # return [1,1,1]
        return self.k_b * np.e ** ((-(dij - self.d0) ** 2) / self.omega)

    def quadraticize(self):
        # TODO:
        pass


class GammaACost(Cost):
    def __init__(self, x_dims, n_dims, D=100.0, d1=80.0, k_g_a=1.0):
        self.x_dims = x_dims
        self.n_dims = n_dims
        self.n_agents = len(x_dims)

        self.D = D
        self.d1 = d1
        self.k_g_a = k_g_a

        self.mu_g_a = np.pi / (D - d1)
        self.v_g_a = -self.mu_g_a * d1

    def __call__(self, x):
        # TODO: Make sure this is correct
        if len(self.x_dims) == 1:
            return 0.0

        # Try to vectorize the distance computation if possible.
        if len(set(self.n_dims)) == 1:
            distances = compute_pairwise_distance(x, self.x_dims)

        # Otherwise, compute the distance to the lower of the i and j number of
        # dimensions.
        else:
            distances = compute_pairwise_distance_nd(
                x.reshape(1, -1), self.x_dims, self.n_dims
            )

        dij = distances
        # print('dij: ', dij)

        gamma_a = np.zeros(dij.shape)
        for i in range(0, self.n_agents):
            if 0 <= dij[0, i] <= self.d1:
                gamma_a[0, i] = self.k_g_a
            elif self.d1 < dij[0, i] <= self.D:
                gamma_a[0, i] = (self.k_g_a/2.0) * (1.0 + np.cos(self.mu_g_a*dij[0, i] + self.v_g_a))
            else:
                gamma_a[0, i] = 0.0

        return gamma_a

    def quadraticize(self):
        # TODO:
        pass


class GammaBCost(Cost):
    def __init__(self, x_dims, n_dims, d_min=0.1, d_max=0.5, k_g_b=1.0):
        self.x_dims = x_dims
        self.n_dims = n_dims
        self.n_agents = len(x_dims)

        self.d_min = d_min
        self.d_max = d_max
        self.k_g_b = k_g_b
        self.mu_g_b = np.pi / (d_max - d_min)
        self.v_g_b = -self.mu_g_b * d_min

    def __call__(self, x, o_k):
        # print("Here")
        n_d = 2
        n_states = self.x_dims[0]
        pair_inds = np.array(list(itertools.combinations(range(self.n_agents), 2)))
        dijks = np.zeros((pair_inds.shape[0]))
        idx_o = 0
        idx_ij = 0
        gamma_b = np.zeros(dijks.shape)
        idx = 0
        for (i, j) in pair_inds:
            dijk = compute_dijk(x[i*n_states:i*n_states+n_d], x[j*n_states:j*n_states+n_d], o_k)
            # print("idx: ", dijk)
            if dijk <= self.d_min:
                gamma_b[idx] = 0.0
            elif self.d_min <= dijk <= self.d_max:
                gamma_b[idx] = (self.k_g_b / 2.0) * (1.0 - np.cos(self.mu_g_b * dijk + self.v_g_b))
            else:  # dijk > d_max_o
                gamma_b[idx] = self.k_g_b
            idx += 1

        return gamma_b

    def quadraticize(self):
        # TODO
        pass


class GammaCost(Cost):
    def __init__(self, x_dims, n_dims, obstacles):
        self.x_dims = x_dims
        self.n_dims = n_dims
        self.n_agents = len(x_dims)

        self.obstacles = obstacles
        self.gamma_a_cost = GammaACost(x_dims, n_dims)
        self.gamma_b_cost = GammaBCost(x_dims, n_dims)

    def __call__(self, x, compute_L=False):
        # TODO: Need dij and dijk, can we modify compute_pairwise... to also compute the dijk?
        # dijks = compute_pairwise_obstacle_distance(x, self.obstacles, self.x_dims)
        gamma_ij = self.gamma_a_cost(x)
        # print([(g**-1)-1 for g in gamma_ij[0]])
        if self.obstacles:
            for o_k in self.obstacles:
                gamma_ij *= self.gamma_b_cost(x, o_k)
        if compute_L:
            return gamma_ij
        # return gamma_ij
        # print([((g+1e-6)**-1) for g in gamma_ij[0]])
        return sum([-g + 1 for g in gamma_ij[0]])
        # g = sum(1.0 - gamma_ij)
        # return sum(1.0 - gamma_ij[0])

    def quadraticize(self,x):
        return quad_los_finite_diff(self, x)
class PDLCost(Cost):
    def __init__(self, x_dims, n_dims, obstacles):
        self.x_dims = x_dims
        self.n_dims = n_dims
        self.n_agents = len(x_dims)
        self.obstacles = obstacles
        self.alpha_cost = AlphaCost(x_dims, n_dims, obstacles)
        self.beta_cost = BetaCost(x_dims, n_dims)
        self.gamma_cost = GammaCost(x_dims, n_dims, obstacles)

        self.V_A = lambda A: 1.0 / (A + 0.01)

    def __call__(self, x, compute_L=False):
        # # State and set of objects
        # # Compute alpha
        # alpha = self.alpha_cost(x, compute_L)
        # # Compute beta
        # beta = self.beta_cost(x)
        # # Compute gamma
        # gamma = self.gamma_cost(x, compute_L)
        #
        # if compute_L:
        #     return (alpha * beta * gamma)
        #
        # return sum(self.V_A(alpha * beta * gamma)[0])
        # State and set of objects
        # Compute alpha
        alpha = self.alpha_cost(x, compute_L)
        # Compute beta
        # beta = self.beta_cost(x)
        # Compute gamma
        gamma = self.gamma_cost(x, compute_L)

        if compute_L:
            return (alpha *  gamma)

        return sum(self.V_A(alpha * gamma)[0])

    def quadraticize(self, x):
        L_x, L_xx = quad_los_finite_diff(self, x)
        return L_x, L_xx


class LambdaCost(Cost):
    def __init__(self, x_dims, n_dims, obstacles, l2_min=0.2):
        self.x_dims = x_dims
        self.n_dims = n_dims
        self.n_agents = len(x_dims)
        self.obstacles = obstacles

        self.l2_min = l2_min
        self.v2 = np.array([])
        # self.V_l = lambda l2: 1.0 / (l2 - l2_min)
        # self.V_l = lambda l2: (l2)**-1
        self.V_l = lambda l2: -1 * min(1.0,l2) + 1


    def __call__(self, x):

        # Compute l2 estimate (globally provided for now)
        # Compute v2 estimate (globally provided for now)
        L = self.compute_L(x)
        # print(L)
        eigs = np.linalg.eig(L)
        # print('e: ', np.linalg.eigvals(L))
        # print('eigs: ', eigs)
        idx = eigs[0].argsort()
        # print('idx: ', idx)
        l2 = eigs[0][idx][1]
        # print('l2:  ', l2)
        # if l2 < 0.0:
        #     print('l2:  ', l2)
        #     input()
        v2 = eigs[1][:, idx][1]
        # print('v2: ', v2)
        return self.V_l(l2)

    def compute_L(self, x):
        # TODO
        # X: Composite state of the system as 1x(N*n_agents) matrix
        x_dim = self.x_dims[0]
        aij_cost = PDLCost([x_dim]*self.n_agents, [2]*self.n_agents, self.obstacles)
        aij = aij_cost(x.copy(), compute_L=True)[0]
        A = np.zeros((self.n_agents, self.n_agents))
        d = np.zeros(A.shape)
        j = 0
        for i in range(0, self.n_agents):
            if len(A[i, i+1:] > 0):
                a = aij[j:j+len(A[i, i+1:])]
                A[i, i+1:] = a
                j += len(A[i, i+1:])

        A += A.T
        for i in range(0, self.n_agents):
            d[i, i] = sum(A[i, :])
        # print("A: ", A)
        # print("d: ", d)
        # print("L: ", (d - A).shape)
        return d - A

    def quadraticize(self, x):
        L_x, L_xx = quad_los_finite_diff(self, x)
        # print()
        # print("L_x: ", L_xx)
        return L_x, L_xx
        # x: agent staties
        # A: Adjacency matrix (alpha_ij, beta_ij, gamma_ij)
        # S: Sensing neighbors list for each agent
        # nx = sum(self.x_dims)
        # nx_per_agent = self.x_dims[0]
        #
        # pair_inds = np.array(list(itertools.combinations(range(self.n_agents), 2)))
        #
        # L_x = np.zeros((nx))
        # L_xx = np.zeros((nx, nx))
        #
        # dV_ldl2 = np.zeros((nx_per_agent))
        #
        # for i in range(0, self.n_agents):
        #     xi = x[i*nx_per_agent:i*nx_per_agent+nx_per_agent]
        #     L_xi = np.zeros((nx_per_agent))
        #     for j in S[i]:
        #         xj = x[j*nx_per_agent:j*nx_per_agent+nx_per_agent]
        #         xij = xi - xj
        #
        #         dAhkdijs = np.zeros((nx_per_agent))
        #         for (h, k) in pair_inds:
        #             if (h != i and k != j) or (k not in S[i]) or (k not in S[j]):
        #                 # Do nothing
        #                 continue
        #
        #             # dAhkdijs += dAhkdij * (self.v2[h] - self.v2[k])**2
        #
        #         L_xi += dV_ldl2 * dAhkdijs
        #
        #     for j in range(0,len(self.obstacles)):
        #         oj = self.obstacles[j]
        #
        #         # TODO: for now just leave at 2
        #         xioj = xi[i*nx_per_agent:i*nx_per_agent+2] - oj
        #
        #         dAhkdiojs = np.zeros((nx_per_agent))
        #         for (h,k) in pair_inds:


        pass


class ConnectivityMaintenanceGameCost(Cost):
    def __init__(self, x_dims, n_dims, reference_costs, obstacles, proximity_cost=None, obst_cost=None):
        self.obstacles = obstacles
        self.ref_costs = reference_costs
        self.prox_cost = proximity_cost
        self.obst_cost = obst_cost

        self.lambda_cost = LambdaCost(x_dims, n_dims, obstacles)
        self.pdl_cost = PDLCost(x_dims, n_dims, obstacles)

        self.REF_WEIGHT = 1.0
        self.OBST_WEIGHT = 200.0
        self.PROX_WEIGHT = 200.0
        self.W1 = 200.0

        self.x_dims = [ref_cost.x_dim for ref_cost in self.ref_costs]
        self.u_dims = [ref_cost.u_dim for ref_cost in self.ref_costs]
        self.ids = [ref_cost.id for ref_cost in self.ref_costs]
        self.n_agents = len(reference_costs)

    @property
    def xf(self):
        return np.concatenate([ref_cost.xf for ref_cost in self.ref_costs])
    def __call__(self, x, u, terminal=False, L=None):
        # Reference total same as in GameCost
        ref_total = 0.0
        lambda_total = 0.0
        pdl_total = 0.0
        for ref_cost, xi, ui in zip(
            self.ref_costs,
            split_agents_gen(x, self.x_dims),
            split_agents_gen(u, self.u_dims),
        ):
            ref_total += ref_cost(xi, ui, terminal)
            # l_cost = self.lam

        lambda_total += self.lambda_cost(x)
        # print()
        # print("l_tot: ", lambda_total)

        # pdl_total = self.pdl_cost(x, self.obstacles)
        # print('ref: ', ref_total)
        # print('lam: ', lambda_total)
        # print('pdl: ', pdl_total)

        return self.PROX_WEIGHT * self.prox_cost(x) + self.OBST_WEIGHT * self.obst_cost(x) + self.REF_WEIGHT * ref_total + self.W1 * lambda_total

    def quadraticize(self, x, u, terminal=False, L=None):
        L_xs, L_us = [], []
        L_xxs, L_uus, L_uxs = [], [], []
        L_x_ls, L_xx_ls = [], []

        # Compute agent quadraticizations in individual state spaces.
        for ref_cost, xi, ui in zip(
                self.ref_costs,
                split_agents_gen(x, self.x_dims),
                split_agents_gen(u, self.u_dims),
        ):
            L_xi, L_ui, L_xxi, L_uui, L_uxi = ref_cost.quadraticize(
                xi.flatten(), ui.flatten(), terminal
            )
            L_xs.append(L_xi)
            L_us.append(L_ui)
            L_xxs.append(L_xxi)
            L_uus.append(L_uui)
            L_uxs.append(L_uxi)

            # L_x_l, L_xx_l = self.lambda_cost.quadraticize(x)
            # L_x_ls.append(L_x_l)
            # L_xx_ls.append(L_xx_l)

        L_x = self.REF_WEIGHT * np.hstack(L_xs)
        # L_x = self.REF_WEIGHT * np.hstack(L_xs) + self.W1 * np.sum(L_x_ls)
        L_u = self.REF_WEIGHT * np.hstack(L_us)
        L_xx = (self.REF_WEIGHT * uniform_block_diag(*L_xxs))
        # L_xx = (self.REF_WEIGHT * uniform_block_diag(*L_xxs) + self.W1 * np.sum(L_xx_ls))
        L_uu = self.REF_WEIGHT * uniform_block_diag(*L_uus)
        L_ux = self.REF_WEIGHT * uniform_block_diag(*L_uxs)

        if self.n_agents > 1:
            L_x_prox, L_xx_prox = self.prox_cost.quadraticize(x)
            L_x += self.PROX_WEIGHT * L_x_prox
            L_xx += self.PROX_WEIGHT * L_xx_prox

            L_x_obst, L_xx_obst = self.obst_cost.quadraticize(x)
            L_x += self.OBST_WEIGHT * L_x_obst
            L_xx += self.OBST_WEIGHT * L_xx_obst

            L_x_l2, L_xx_l2 = self.lambda_cost.quadraticize(x)
            L_x += self.W1 * L_x_l2
            L_xx += self.W1 * L_xx_l2

        return L_x, L_u, L_xx, L_uu, L_ux

    def split(self, graph):
        """Split this model into sub game-costs dictated by the interaction graph"""
        # TODO
        # # Assume all states and radii are the same between agents.
        # n_states = self.ref_costs[0].x_dim
        # radius = self.prox_cost.radius
        # n_dims = self.prox_cost.n_dims
        #
        # game_costs = []
        # for prob_ids in graph.values():
        #
        #     goal_costs_i = []
        #     n_dims_i = []
        #     for n_dim, ref_cost in zip(n_dims, self.ref_costs):
        #         if ref_cost.id in prob_ids:
        #             goal_costs_i.append(ref_cost)
        #             n_dims_i.append(n_dim)
        #
        #     prox_cost_i = ProximityCost([n_states] * len(prob_ids), radius, n_dims_i)
        #     game_costs.append(GameCost(goal_costs_i, prox_cost_i))
        #
        # return game_costs
        pass

class ConnectivityMaintenanceGameCost2(Cost):
    def __init__(self, x_dims, n_dims, reference_costs, proximity_cost, obstacles):
        self.obstacles = obstacles
        self.ref_costs = reference_costs
        self.prox_cost = proximity_cost

        self.lambda_cost = LambdaCost(x_dims, n_dims, obstacles)
        self.pdl_cost = PDLCost(x_dims, n_dims, obstacles)
        self.obstacles = obstacles
        self.alpha_cost = AlphaCost(x_dims, n_dims, obstacles)
        self.beta_cost = BetaCost(x_dims, n_dims)
        self.gamma_cost = GammaCost(x_dims, n_dims, obstacles)

        self.REF_WEIGHT = 1.0
        self.W1 = 1.0
        self.W2 = 1.0

        self.x_dims = [ref_cost.x_dim for ref_cost in self.ref_costs]
        self.u_dims = [ref_cost.u_dim for ref_cost in self.ref_costs]
        self.ids = [ref_cost.id for ref_cost in self.ref_costs]
        self.n_agents = len(reference_costs)

    @property
    def xf(self):
        return np.concatenate([ref_cost.xf for ref_cost in self.ref_costs])
    def __call__(self, x, u, terminal=False, L=None):
        # Reference total same as in GameCost
        ref_total = 0.0
        lambda_total = 0.0
        pdl_total = 0.0
        for ref_cost, xi, ui in zip(
            self.ref_costs,
            split_agents_gen(x, self.x_dims),
            split_agents_gen(u, self.u_dims),
        ):
            ref_total += ref_cost(xi, ui, terminal)
            # l_cost = self.lam
        # lambda_total += self.lambda_cost(x)
        # print()
        # print("l_tot: ", lambda_total)

        pdl_total = self.pdl_cost(x)
        print('ref: ', ref_total)
        # print('lam: ', lambda_total)
        print('pdl: ', pdl_total)
        # a = self.alpha_cost(x)
        # g = self.gamma_cost(x)
        #
        # print("a: ", a)
        # print("g: ", g)

        return self.REF_WEIGHT * ref_total + self.W1 * pdl_total + self.W2 * self.prox_cost(x)# + self.W1 * a + self.W2 * g

    def quadraticize(self, x, u, terminal=False, L=None):
        L_xs, L_us = [], []
        L_xxs, L_uus, L_uxs = [], [], []
        L_x_as, L_xx_as = [], []
        L_x_gs, L_xx_gs = [], []

        # Compute agent quadraticizations in individual state spaces.
        for ref_cost, xi, ui in zip(
                self.ref_costs,
                split_agents_gen(x, self.x_dims),
                split_agents_gen(u, self.u_dims),
        ):
            L_xi, L_ui, L_xxi, L_uui, L_uxi = ref_cost.quadraticize(
                xi.flatten(), ui.flatten(), terminal
            )
            L_xs.append(L_xi)
            L_us.append(L_ui)
            L_xxs.append(L_xxi)
            L_uus.append(L_uui)
            L_uxs.append(L_uxi)

            # L_x_a, L_xx_a = self.alpha_cost.quadraticize(x)
            # L_x_as.append(L_x_a)
            # L_xx_as.append(L_xx_a)
            #
            # L_x_g, L_xx_g = self.gamma_cost.quadraticize(x)
            # L_x_gs.append(L_x_g)
            # L_xx_gs.append(L_xx_g)

        L_x = self.REF_WEIGHT * np.hstack(L_xs)# + self.W1 * np.sum(L_x_as) + self.W2 * np.sum(L_x_gs)
        L_u = self.REF_WEIGHT * np.hstack(L_us)
        L_xx = self.REF_WEIGHT * uniform_block_diag(*L_xxs)#+ self.W1 * np.sum(L_xx_as) + self.W2 * np.sum(L_xx_gs)
        L_uu = self.REF_WEIGHT * uniform_block_diag(*L_uus)
        L_ux = self.REF_WEIGHT * uniform_block_diag(*L_uxs)

        if self.n_agents > 1:
            L_x_p, L_xx_p = self.pdl_cost.quadraticize(x)
            L_x += self.W1 * L_x_p
            L_xx += self.W1 * L_xx_p

            L_x_prox, L_xx_prox = self.prox_cost.quadraticize(x)
            L_x += self.W2 * L_x_prox
            L_xx += self.W2 * L_xx_prox
        #     L_x_a, L_xx_a = self.alpha_cost.quadraticize(x)
        #     L_x += self.W1 * L_x_a
        #     L_xx += self.W1 * L_xx_a
        #
        #     L_x_g, L_xx_g = self.gamma_cost.quadraticize(x)
        #     L_x += self.W2 * L_x_g
        #     L_xx += self.W2 * L_xx_g

        return L_x, L_u, L_xx, L_uu, L_ux

    def split(self, graph):
        """Split this model into sub game-costs dictated by the interaction graph"""
        # TODO
        # # Assume all states and radii are the same between agents.
        # n_states = self.ref_costs[0].x_dim
        # radius = self.prox_cost.radius
        # n_dims = self.prox_cost.n_dims
        #
        # game_costs = []
        # for prob_ids in graph.values():
        #
        #     goal_costs_i = []
        #     n_dims_i = []
        #     for n_dim, ref_cost in zip(n_dims, self.ref_costs):
        #         if ref_cost.id in prob_ids:
        #             goal_costs_i.append(ref_cost)
        #             n_dims_i.append(n_dim)
        #
        #     prox_cost_i = ProximityCost([n_states] * len(prob_ids), radius, n_dims_i)
        #     game_costs.append(GameCost(goal_costs_i, prox_cost_i))
        #
        # return game_costs
        pass

class GameCost(Cost):
    def __init__(self, reference_costs, proximity_cost=None, obst_cost=None):

        if not proximity_cost:

            def proximity_cost(_):
                return 0.0

        self.ref_costs = reference_costs
        # print(self.ref_costs)
        self.prox_cost = proximity_cost
        self.obst_cost = obst_cost

        self.REF_WEIGHT = 1.0
        self.OBST_WEIGHT = 10.0
        self.PROX_WEIGHT = 200.0

        self.x_dims = [ref_cost.x_dim for ref_cost in self.ref_costs]
        self.u_dims = [ref_cost.u_dim for ref_cost in self.ref_costs]
        self.ids = [ref_cost.id for ref_cost in self.ref_costs]
        self.n_agents = len(reference_costs)

    @property
    def xf(self):
        return np.concatenate([ref_cost.xf for ref_cost in self.ref_costs])

    def __call__(self, x, u, terminal=False):
        ref_total = 0.0
        for ref_cost, xi, ui in zip(
            self.ref_costs,
            split_agents_gen(x, self.x_dims),
            split_agents_gen(u, self.u_dims),
        ):
            ref_total += ref_cost(xi, ui, terminal)

        return self.PROX_WEIGHT * self.prox_cost(x) + self.OBST_WEIGHT * self.obst_cost(x) + self.REF_WEIGHT * ref_total

    def quadraticize(self, x, u, terminal=False):
        L_xs, L_us = [], []
        L_xxs, L_uus, L_uxs = [], [], []

        # Compute agent quadraticizations in individual state spaces.
        for ref_cost, xi, ui in zip(
            self.ref_costs,
            split_agents_gen(x, self.x_dims),
            split_agents_gen(u, self.u_dims),
        ):
            L_xi, L_ui, L_xxi, L_uui, L_uxi = ref_cost.quadraticize(
                xi.flatten(), ui.flatten(), terminal
            )
            L_xs.append(L_xi)
            L_us.append(L_ui)
            L_xxs.append(L_xxi)
            L_uus.append(L_uui)
            L_uxs.append(L_uxi)

        L_x = self.REF_WEIGHT * np.hstack(L_xs)
        L_u = self.REF_WEIGHT * np.hstack(L_us)
        L_xx = self.REF_WEIGHT * uniform_block_diag(*L_xxs)
        L_uu = self.REF_WEIGHT * uniform_block_diag(*L_uus)
        L_ux = self.REF_WEIGHT * uniform_block_diag(*L_uxs)

        # Incorporate coupling costs in full cartesian state space.
        if self.n_agents > 1:
            L_x_prox, L_xx_prox = self.prox_cost.quadraticize(x)
            L_x += self.PROX_WEIGHT * L_x_prox
            L_xx += self.PROX_WEIGHT * L_xx_prox

            L_x_obst, L_xx_obst = self.obst_cost.quadraticize(x)
            L_x += self.OBST_WEIGHT * L_x_obst
            L_xx += self.OBST_WEIGHT * L_xx_obst

        # print("L_x: ", L_x.shape)
        # print("L_u: ", L_u.shape)
        # print("L_xx: ", L_xx.shape)
        # print("L_uu: ", L_uu.shape)
        # print("L_ux: ", L_ux.shape)
        # input()

        return L_x, L_u, L_xx, L_uu, L_ux

    def split(self, graph):
        """Split this model into sub game-costs dictated by the interaction graph"""

        # Assume all states and radii are the same between agents.
        n_states = self.ref_costs[0].x_dim
        radius = self.prox_cost.radius
        n_dims = self.prox_cost.n_dims

        game_costs = []
        for prob_ids in graph.values():

            goal_costs_i = []
            n_dims_i = []
            for n_dim, ref_cost in zip(n_dims, self.ref_costs):
                if ref_cost.id in prob_ids:
                    goal_costs_i.append(ref_cost)
                    n_dims_i.append(n_dim)

            prox_cost_i = ProximityCost([n_states] * len(prob_ids), radius, n_dims_i)
            game_costs.append(GameCost(goal_costs_i, prox_cost_i))

        return game_costs

    def __repr__(self):
        ids = [ref_cost.id for ref_cost in self.ref_costs]
        return f"GameCost(\n\tids: {ids},\n\tprox_cost: {self.prox_cost}\n)"


class GameCost2(Cost):
    def __init__(self, reference_costs, proximity_cost=None, obst_cost=None, gamma_cost=None):

        if not proximity_cost:
            def proximity_cost(_):
                return 0.0

        self.ref_costs = reference_costs
        # print(self.ref_costs)
        self.prox_cost = proximity_cost
        self.obst_cost = obst_cost
        self.gamma_cost = gamma_cost

        self.REF_WEIGHT = 1.0
        self.OBST_WEIGHT = 200.0
        self.PROX_WEIGHT = 200.0
        self.G_WEIGHT = 200.0

        self.x_dims = [ref_cost.x_dim for ref_cost in self.ref_costs]
        self.u_dims = [ref_cost.u_dim for ref_cost in self.ref_costs]
        self.ids = [ref_cost.id for ref_cost in self.ref_costs]
        self.n_agents = len(reference_costs)

    @property
    def xf(self):
        return np.concatenate([ref_cost.xf for ref_cost in self.ref_costs])

    def __call__(self, x, u, terminal=False):
        ref_total = 0.0
        for ref_cost, xi, ui in zip(
                self.ref_costs,
                split_agents_gen(x, self.x_dims),
                split_agents_gen(u, self.u_dims),
        ):
            ref_total += ref_cost(xi, ui, terminal)

        return (self.PROX_WEIGHT * self.prox_cost(x) + self.OBST_WEIGHT * self.obst_cost(x) +
                self.G_WEIGHT * self.gamma_cost(x) + self.REF_WEIGHT * ref_total)

    def quadraticize(self, x, u, terminal=False):
        L_xs, L_us = [], []
        L_xxs, L_uus, L_uxs = [], [], []

        # Compute agent quadraticizations in individual state spaces.
        for ref_cost, xi, ui in zip(
                self.ref_costs,
                split_agents_gen(x, self.x_dims),
                split_agents_gen(u, self.u_dims),
        ):
            L_xi, L_ui, L_xxi, L_uui, L_uxi = ref_cost.quadraticize(
                xi.flatten(), ui.flatten(), terminal
            )
            L_xs.append(L_xi)
            L_us.append(L_ui)
            L_xxs.append(L_xxi)
            L_uus.append(L_uui)
            L_uxs.append(L_uxi)

        L_x = self.REF_WEIGHT * np.hstack(L_xs)
        L_u = self.REF_WEIGHT * np.hstack(L_us)
        L_xx = self.REF_WEIGHT * uniform_block_diag(*L_xxs)
        L_uu = self.REF_WEIGHT * uniform_block_diag(*L_uus)
        L_ux = self.REF_WEIGHT * uniform_block_diag(*L_uxs)

        # Incorporate coupling costs in full cartesian state space.
        if self.n_agents > 1:
            L_x_prox, L_xx_prox = self.prox_cost.quadraticize(x)
            L_x += self.PROX_WEIGHT * L_x_prox
            L_xx += self.PROX_WEIGHT * L_xx_prox

            L_x_obst, L_xx_obst = self.obst_cost.quadraticize(x)
            L_x += self.OBST_WEIGHT * L_x_obst
            L_xx += self.OBST_WEIGHT * L_xx_obst

            L_x_g, L_xx_g = self.gamma_cost.quadraticize(x)
            L_x += self.G_WEIGHT * L_x_g
            L_xx += self.G_WEIGHT * L_xx_g

        # print("L_x: ", L_x.shape)
        # print("L_u: ", L_u.shape)
        # print("L_xx: ", L_xx.shape)
        # print("L_uu: ", L_uu.shape)
        # print("L_ux: ", L_ux.shape)
        # input()

        return L_x, L_u, L_xx, L_uu, L_ux

    def split(self, graph):
        """Split this model into sub game-costs dictated by the interaction graph"""

        # Assume all states and radii are the same between agents.
        n_states = self.ref_costs[0].x_dim
        radius = self.prox_cost.radius
        n_dims = self.prox_cost.n_dims

        game_costs = []
        for prob_ids in graph.values():

            goal_costs_i = []
            n_dims_i = []
            for n_dim, ref_cost in zip(n_dims, self.ref_costs):
                if ref_cost.id in prob_ids:
                    goal_costs_i.append(ref_cost)
                    n_dims_i.append(n_dim)

            prox_cost_i = ProximityCost([n_states] * len(prob_ids), radius, n_dims_i)
            game_costs.append(GameCost(goal_costs_i, prox_cost_i))

        return game_costs

    def __repr__(self):
        ids = [ref_cost.id for ref_cost in self.ref_costs]
        return f"GameCost(\n\tids: {ids},\n\tprox_cost: {self.prox_cost}\n)"



def compute_dijk(x_i, x_j, o_k):
    x1, y1 = x_i[0], x_i[1]
    x2, y2 = x_j[0], x_j[1]
    x3, y3 = o_k[0], o_k[1]
    dx, dy = x2 - x1, y2 - y1
    det = dx*dx + dy*dy
    a = (dy*(y3-y1) + dx*(x3-x1))/det
    sijk = np.array([x1+a*dx, y1+a*dy])
    if is_between(x_i, x_j, sijk):
        return np.linalg.norm(np.cross((o_k - x_j), (o_k - x_i))) / np.linalg.norm((x_j - x_i))
    else:
        if np.linalg.norm(x_i - sijk) < np.linalg.norm(x_j - sijk):
            return np.linalg.norm(x_i - sijk)
        else:
            return np.linalg.norm(x_j - sijk)


def is_between(a,b,c):
    return -0.01 < (np.linalg.norm(a-c) + np.linalg.norm(c-b) - np.linalg.norm(a-b)) < 0.01


def compute_pairwise_obstacle_distance(X, O, x_dims, n_d=2):
    # assert len(set(x_dims)) == 1
    if not O:
        return np.array([])
    n_agents = len(x_dims)
    n_states = x_dims[0]

    if n_agents == 1:
        raise ValueError("Can't compute pairwise obstacle distance for one agent")

    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    # O = np.array([[1.0, 1.0], [2, 2]])
    dijks = np.zeros((len(O), pair_inds.shape[0]))
    idx_o = 0
    idx_ij = 0
    # O = np.array([[1.0, 1.0]])
    for o in O:
        for (i, j) in pair_inds:
            dijks[idx_o, idx_ij] = compute_dijk(X[i*n_states:i*n_states+n_d], X[j*n_states:j*n_states+n_d], o)
            idx_ij += 1
        idx_ij = 0
        idx_o += 1
    # print('dijks: ', dijks)
    return dijks

def quadraticize_distance(point_a, point_b, radius, n_d):
    """Quadraticize the distance between two points thresholded by a radius
       in either 2 or 3 dimensions returning the n_d x 1 jacobian and
       n_d x n_d hessian.

    NOTE: we assume that the states are organized in matrix form as [x, y, z, ...].
    NOTE: this still works in two dimensions since the default z value for the
          point class is 0.
    """

    assert point_a.ndim == point_b.ndim

    L_x = np.zeros((3))
    L_xx = np.zeros((3, 3))

    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    dz = point_a.z - point_b.z
    distance = np.sqrt(dx * dx + dy * dy + dz * dz)

    if distance > radius:
        return L_x[:n_d], L_xx[:n_d, :n_d]

    L_x = 2 * (distance - radius) / distance * np.array([dx, dy, dz])

    cross_factors = (
        2
        * radius
        / np.sqrt(
            (point_a.hypot2() + point_b.hypot2())
            - 2
            * (point_a.x * point_b.x + point_a.y * point_b.y + point_a.z * point_b.z)
        )
        ** 3
    )

    L_xx[np.diag_indices(3)] = (
        2 * radius * np.array([dx, dy, dz]) ** 2 / distance**3
        - 2 * radius / distance
        + 2
    )

    L_xx[np.tril_indices(3, -1)] = L_xx[np.triu_indices(3, 1)] = (
        np.array([dx * dy, dx * dz, dy * dz]) * cross_factors
    )

    return L_x[:n_d], L_xx[:n_d, :n_d]


def quadraticize_finite_difference(cost, x, u, terminal=False, jac_eps=None):
    """Finite difference quadraticized cost

    NOTE: deprecated in favor of automatic differentiation in lieu of speed and
    consistency.
    """
    if not jac_eps:
        jac_eps = np.sqrt(np.finfo(float).eps)
    hess_eps = np.sqrt(jac_eps)

    n_x = x.shape[0]
    n_u = u.shape[0]

    def Lx(x, u):
        return approx_fprime(x, lambda x: cost(x, u, terminal), jac_eps)

    def Lu(x, u):
        return approx_fprime(u, lambda u: cost(x, u, terminal), jac_eps)

    L_xx = np.vstack(
        [approx_fprime(x, lambda x: Lx(x, u)[i], hess_eps) for i in range(n_x)]
    )

    L_uu = np.vstack(
        [approx_fprime(u, lambda u: Lu(x, u)[i], hess_eps) for i in range(n_u)]
    )

    L_ux = np.vstack(
        [approx_fprime(x, lambda x: Lu(x, u)[i], hess_eps) for i in range(n_u)]
    )

    return Lx(x, u), Lu(x, u), L_xx, L_uu, L_ux


def quad_los_finite_diff(cost, x, jac_eps=False):
    if not jac_eps:
        jac_eps = np.sqrt(np.finfo(float).eps)
    hess_eps = np.sqrt(jac_eps)

    def Lx(x):
        return approx_fprime(x, lambda x: cost(x), jac_eps)

    L_xx = approx_fprime(x, lambda x: Lx(x), hess_eps)

    return Lx(x), L_xx
