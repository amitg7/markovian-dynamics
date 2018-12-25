import numpy as np
from markoviandynamics.symbolic import SymbolicDiscreteSystemArrhenius
from numpy.testing import assert_equal


def test_discrete_system():
    system = SymbolicDiscreteSystemArrhenius(3)
    energies = np.random.rand(30).reshape(3, 10)
    temperatures = np.random.rand(10)

    eq = system.equilibrium_lambdified(temperatures, *energies)
    for i in range(np.size(temperatures)):
        assert_equal(eq[:,:,i], system.equilibrium_lambdified(temperatures[i], *energies[:,i]))

    eq = system.equilibrium_lambdified(temperatures, *energies[:,0])
    for i in range(np.size(temperatures)):
        assert_equal(eq[:,:,i], system.equilibrium_lambdified(temperatures[i], *energies[:,0]))

    eq = system.equilibrium_lambdified(temperatures[0], *energies)
    for i in range(np.size(energies.shape[1])):
        assert_equal(eq[:,:,i], system.equilibrium_lambdified(temperatures[0], *energies[:,i]))
