import copy

import numpy as np
from GooseEPM import laplace_propagator
from GooseEPM import SystemAthermal

try:
    import matplotlib.pyplot as plt

    plot = True
except ImportError:
    plot = False

L = 100
system = SystemAthermal(
    *laplace_propagator(),
    sigmay_mean=np.ones([L, L]),
    sigmay_std=0.3 * np.ones([L, L]),
    seed=0,
    init_random_stress=True,
    init_relax=True,
)

assert system.propogator_follows_conventions("stress")
base = copy.deepcopy(system)


sigma = np.linspace(0.55, 1, 20)
gammadot = np.zeros_like(sigma)

for i in range(len(sigma)):
    system = copy.deepcopy(base)
    system.sigmabar = sigma[i]

    # preparation
    system.makeAthermalFailureSteps(3 * system.size)

    # measurement
    epsp0 = np.mean(system.epsp)
    t = system.t
    system.makeAthermalFailureSteps(system.size)

    if np.sum(system.nfails) < 4 * system.size:
        continue

    gammadot[i] = (np.mean(system.epsp) - epsp0) / (system.t - t)


if plot:
    fig, axes = plt.subplots(ncols=2, figsize=(8 * 2, 6))

    ax = axes[0]
    ax.plot(gammadot, sigma, marker=".")
    ax.set_xlabel(r"$\dot\gamma$")
    ax.set_ylabel(r"$\sigma$")

    ax = axes[1]
    cax = ax.imshow(system.epsp, interpolation="nearest")

    cbar = fig.colorbar(cax, aspect=10)
    cbar.set_label(r"$\gamma_p$")

    plt.show()
