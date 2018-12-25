import sympy as sp
import numpy as np


class SymbolicDiscreteSystem:
    """
    Symbolic representation of discrete system.
    """
    def __init__(self, N):
        """
        Parameters
        ----------
        N : int
            Number of states.
        """
        self.N = N

        # Symbols
        self.T = sp.symbols('T', real=True)
        self.E_i = sp.symbols('E_1:%d' % (self.N + 1), real=True)

        # Partition function
        self.Z = np.sum([sp.exp(-E / self.T) for E in self.E_i])

        # Equilibrium
        self._eq = sp.Matrix([sp.exp(-E / self.T) for E in self.E_i]) / self.Z

        # Lambdified equilibrium probability distribution
        self.equilibrium_lambdified = self._equilibrium_lambdify()

    def _equilibrium_lambdify(self):
        return sp.lambdify([self.T, *self.E_i], self._eq)

    def equilibrium(self, energies=None, temperature=None):
        """
        Return the equilibrium probability distribution.

        If additional parameters are given, the expression returned will be substituted with the given values.

        Parameters
        ----------
        energies : (N,) float array or sequence
            Energies to substitute.
        temperature : float or (N,) array
            Temperature to substitute.

        Returns
        -------
        expr : (N, 1) array
            Equilibrium probability distribution.
        """
        expr = self._eq
        if energies is not None:
            expr = expr.subs(dict(zip(self.E_i, np.array(energies))))
        if temperature is not None:
            if np.size(temperature) == 1:
                expr = expr.subs(self.T, temperature)
            elif expr.free_symbols == {self.T}:
                return np.vstack(sp.lambdify(self.T, expr)(temperature))
        if not expr.free_symbols:
            expr = np.array(expr).astype(np.float64)
        return expr


class SymbolicDiscreteSystemArrhenius(SymbolicDiscreteSystem):
    """
    Symbolic representation of a discrete system with potential energy barriers.
    """
    @staticmethod
    def _barrier_element(i, j):
        if i == j:
            return 0
        return sp.symbols('B_%d%d' % (i + 1, j + 1), real=True)

    def __init__(self, N):
        """
        Parameters
        ----------
        N : int
            Number of states.
        """
        super().__init__(N)

        # Potential energy barriers
        self.B_ij = sp.Matrix(self.N, self.N, self._barrier_element)
