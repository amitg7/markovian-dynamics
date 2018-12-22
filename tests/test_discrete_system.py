import numpy as np
from markoviandynamics.symbolic import SymbolicDiscreteSystemArrhenius
from numpy.testing import assert_equal


def test_discrete_system():
    system = SymbolicDiscreteSystemArrhenius(3)
    energies = np.random.rand(30).reshape(3, 10)
    temperatures = np.random.rand(10)

    eq = system.equilibrium(energies=energies, temperature=temperatures)
    for i in range(np.size(temperatures)):
        assert_equal(eq[:,:,i], system.equilibrium(energies=energies[:,i], temperature=temperatures[i]))

    eq = system.equilibrium(energies=energies[:,0], temperature=temperatures)
    for i in range(np.size(temperatures)):
        assert_equal(eq[:,:,i], system.equilibrium(energies=energies[:,0], temperature=temperatures[i]))

    eq = system.equilibrium(energies=energies, temperature=temperatures[0])
    for i in range(np.size(energies.shape[1])):
        assert_equal(eq[:,:,i], system.equilibrium(energies=energies[:,i], temperature=temperatures[0]))
