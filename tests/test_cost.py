#!/usr/bin/env python

"""Unit testing for the cost module"""

import unittest

import numpy as np

import dpilqr


class TestProximityCost(unittest.TestCase):
    def test_single(self):
        cost = dpilqr.ProximityCost([2], 10.0)([1, 2])
        self.assertAlmostEqual(cost, 0.0)

    def test_call_2(self):
        r = 10.0
        x = np.array([0, 0, 0, 1, 2, 0])
        cost = dpilqr.ProximityCost([3, 3], r)(x)
        expected = (np.hypot(1, 2) - r) ** 2
        self.assertAlmostEqual(cost, expected)

    def test_quadraticize_3(self):
        x = np.arange(9)
        cost = dpilqr.ProximityCost([3, 3, 3], 10.0)
        Lx, _, Lxx, *_ = cost.quadraticize(x)

        u = np.zeros(6)
        Lx_diff, _, Lxx_diff, *_ = dpilqr.quadraticize_finite_difference(
            cost.__call__, x, u, False
        )

        # Approximately validate with finite difference.
        self.assertTrue(np.allclose(Lx, Lx_diff, atol=0.1))
        self.assertTrue(np.allclose(Lxx, Lxx_diff, atol=0.1))


class TestReferenceCost(unittest.TestCase):
    def setUp(self):
        self.n, self.m = 3, 2
        self.xf = np.zeros(self.n)
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.Qf = np.diag([1, 1, 0])

        self.ref_cost = dpilqr.ReferenceCost(self.xf, self.Q, self.R, self.Qf)

        self.x0 = np.random.randint(0, 10, (self.n,))
        self.u = np.random.randint(0, 10, (self.m,))

    def test_call(self):
        expected = np.sum(np.linalg.norm(self.x0) ** 2) + np.sum(
            np.linalg.norm(self.u) ** 2
        )
        self.assertAlmostEqual(expected, self.ref_cost(self.x0, self.u))

    def test_terminal_call(self):
        expected = np.sum(np.linalg.norm(self.x0[:-1]) ** 2)
        self.assertAlmostEqual(expected, self.ref_cost(self.x0, self.u, terminal=True))

    def test_quadraticize(self):
        Q_plus_Q_T = 2 * np.eye(self.n)
        R_plus_R_T = 2 * np.eye(self.m)

        L_x_expect = self.x0.T @ Q_plus_Q_T
        L_u_expect = self.u.T @ R_plus_R_T
        L_xx_expect = Q_plus_Q_T
        L_uu_expect = R_plus_R_T
        L_ux_expect = np.zeros((self.m, self.n))

        L_x, L_u, L_xx, L_uu, L_ux = self.ref_cost.quadraticize(self.x0, self.u)

        self.assertTrue(np.allclose(L_x, L_x_expect))
        self.assertTrue(np.allclose(L_u, L_u_expect))
        self.assertTrue(np.allclose(L_xx, L_xx_expect))
        self.assertTrue(np.allclose(L_uu, L_uu_expect))
        self.assertTrue(np.allclose(L_ux, L_ux_expect))


class TestGameCost(unittest.TestCase):
    def test_single(self):
        xf = np.zeros(4)
        Q = np.eye(4)
        R = np.eye(2)
        player_cost = dpilqr.ReferenceCost(xf, Q, R)
        game_cost = dpilqr.GameCost([player_cost])

        x = np.random.rand(4)
        u = np.random.rand(2)
        cost_expect = np.sum(x**2) + np.sum(u**2)
        self.assertAlmostEqual(player_cost(x, u), game_cost(x, u))
        self.assertAlmostEqual(player_cost(x, u), cost_expect)

        # Approximately validate with finite difference.
        Lx, Lu, Lxx, Luu, _ = game_cost.quadraticize(x, u)
        Lx_diff, Lu_diff, Lxx_diff, Luu_diff, _ = dpilqr.quadraticize_finite_difference(
            game_cost.__call__, x, u
        )

        self.assertTrue(np.allclose(Lx, Lx_diff, atol=0.001))
        self.assertTrue(np.allclose(Lu, Lu_diff, atol=0.001))
        self.assertTrue(np.allclose(Lxx, Lxx_diff, atol=0.1))
        self.assertTrue(np.allclose(Luu, Luu_diff, atol=0.1))

    def test_multi(self):
        xf = np.array([0, 0, -1, 2, 2, -1, 2, 0, -1])
        Q = np.eye(3)
        R = np.eye(2)
        x_dims = [3, 3, 3]
        radius = 5.0
        player_costs = [
            dpilqr.ReferenceCost(xfi, Q, R) for xfi in dpilqr.split_agents(xf, x_dims)
        ]
        prox_cost = dpilqr.ProximityCost(x_dims, radius)
        game_cost = dpilqr.GameCost(player_costs, prox_cost)

        x = 10 * np.random.randn(9)
        u = np.random.randn(6)
        Lx, Lu, Lxx, Luu, _ = game_cost.quadraticize(x, u)
        Lx_diff, Lu_diff, Lxx_diff, Luu_diff, _ = dpilqr.quadraticize_finite_difference(
            game_cost.__call__, x, u
        )

        # NOTE: The accuracy of the numerical differentiation can only get so close.
        self.assertTrue(np.allclose(Lx, Lx_diff, atol=0.001))
        self.assertTrue(np.allclose(Lu, Lu_diff, atol=0.001))
        self.assertTrue(np.allclose(Lxx, Lxx_diff, atol=1.0))
        self.assertTrue(np.allclose(Luu, Luu_diff, atol=1.0))


if __name__ == "__main__":
    unittest.main()
