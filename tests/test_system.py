import copy
import faulthandler
import unittest

import numpy as np
from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal

faulthandler.enable()


class Test_SystemAthermal(unittest.TestCase):
    """
    GooseEPM.SystemAthermal
    """

    def test_propagator(self):
        """
        elshelby_propagator (Python only)
        """

        propagator, _, _ = elshelby_propagator(L=100, imposed="strain")
        self.assertAlmostEqual(propagator[0, 0], -1)
        self.assertAlmostEqual(np.sum(propagator), -1)

        propagator, _, _ = elshelby_propagator(L=100, imposed="stress")
        self.assertAlmostEqual(propagator[0, 0], -1)
        self.assertAlmostEqual(np.sum(propagator), 0)

    def test_imposedStress(self):
        """
        Check that stress is preserved
        """

        L = 61
        sigmabar = 0.5
        system = SystemAthermal(
            *elshelby_propagator(L=L, imposed="stress"),
            sigmay_mean=np.ones([L, L]),
            sigmay_std=np.zeros([L, L]),
            seed=0,
            sigmabar=sigmabar,
            init_random_stress=True,
            init_relax=True,
        )
        self.assertAlmostEqual(system.sigmabar, sigmabar)
        self.assertTrue(system.propogator_follows_conventions("stress"))

        sigmabar = 0.6
        system.sigmabar = sigmabar
        self.assertAlmostEqual(system.sigmabar, sigmabar)

        system.relaxAthermal()
        self.assertAlmostEqual(system.sigmabar, sigmabar)

    def test_imposedStrain(self):
        """
        Check that stress is changed corectly
        """

        L = 61
        system = SystemAthermal(
            *elshelby_propagator(L=L, imposed="strain"),
            sigmay_mean=np.ones([L, L]),
            sigmay_std=np.zeros([L, L]),
            seed=0,
            init_random_stress=True,
            init_relax=True,
        )
        self.assertTrue(system.propogator_follows_conventions("strain"))

        system.shiftImposedShear(direction=1)

        sigma = np.copy(system.sigma)
        i = np.argwhere(system.sigma > system.sigmay).ravel()
        idx = np.ravel_multi_index(i, system.sigma.shape)

        system.spatialParticleFailure(idx)

        dsig = sigma.flat[idx] - system.sigma.flat[idx]
        self.assertAlmostEqual(system.sigmabar, np.mean(sigma) - dsig / sigma.size)

    def test_shiftImposedShear(self):
        """
        shiftImposedShear
        """

        propagator = np.array(
            [
                [0, 0.25, 0],
                [0.25, -1, 0.25],
                [0, 0.25, 0],
            ]
        )

        system = SystemAthermal(
            propagator=propagator,
            distances_rows=np.array([-1, 0, 1]),
            distances_cols=np.array([-1, 0, 1]),
            sigmay_mean=np.ones([5, 5]),
            sigmay_std=np.zeros([5, 5]),
            seed=0,
        )

        system.sigma = 0.1 * np.ones_like(system.sigma)
        self.assertAlmostEqual(system.sigmabar, 0.1)

        system.shiftImposedShear(direction=1)
        self.assertTrue(np.allclose(system.sigma, 1))

        system.shiftImposedShear(direction=-1)
        self.assertTrue(np.allclose(system.sigma, -1))

    def test_relaxAthermal(self):
        """
        relaxAthermal
        """

        propagator = np.array(
            [
                [0, 0.25, 0],
                [0.25, -1, 0.25],
                [0, 0.25, 0],
            ]
        )

        system = SystemAthermal(
            propagator=propagator,
            distances_rows=np.array([-1, 0, 1]),
            distances_cols=np.array([-1, 0, 1]),
            sigmay_mean=np.ones([5, 5]),
            sigmay_std=0.1 * np.ones([5, 5]),
            seed=0,
        )

        self.assertAlmostEqual(system.sigmabar, np.mean(system.sigma))

        # holds only if "sigmay" is sufficiently high compare to "sigma" (true here though)
        self.assertTrue(np.allclose(system.epsp, 0))

        system.shiftImposedShear(direction=1)
        system.relaxAthermal()

        self.assertTrue(np.all(system.epsp >= 0))
        self.assertTrue(np.all(np.abs(system.sigma) < system.sigmay))

    def test_copy(self):
        """
        copy
        """

        propagator = np.array(
            [
                [0, 0.25, 0],
                [0.25, -1, 0.25],
                [0, 0.25, 0],
            ]
        )

        system = SystemAthermal(
            propagator=propagator,
            distances_rows=np.array([-1, 0, 1]),
            distances_cols=np.array([-1, 0, 1]),
            sigmay_mean=np.ones([5, 5]),
            sigmay_std=0.1 * np.ones([5, 5]),
            seed=0,
        )

        t = system.t
        state = system.state
        sigma = np.copy(system.sigma)
        epsp = np.copy(system.epsp)

        mycopy = copy.copy(system)

        system.shiftImposedShear(direction=1)
        system.relaxAthermal()

        self.assertNotEqual(system.t, t)
        self.assertNotEqual(system.state, state)
        self.assertFalse(np.allclose(system.sigma, sigma))
        self.assertFalse(np.allclose(system.epsp, epsp))

        self.assertEqual(mycopy.t, t)
        self.assertEqual(mycopy.state, state)
        self.assertTrue(np.allclose(mycopy.sigma, sigma))
        self.assertTrue(np.allclose(mycopy.epsp, epsp))

        mycopy.shiftImposedShear(direction=1)
        mycopy.relaxAthermal()

        self.assertEqual(mycopy.t, system.t)
        self.assertEqual(mycopy.state, system.state)
        self.assertTrue(np.allclose(mycopy.sigma, system.sigma))
        self.assertTrue(np.allclose(mycopy.epsp, system.epsp))


if __name__ == "__main__":
    unittest.main(verbosity=2)
