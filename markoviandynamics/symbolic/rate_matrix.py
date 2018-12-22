import sympy as sp
import numpy as np
import pickle


class SymbolicRateMatrixArrhenius(sp.Matrix):
    """
    Symbolic representation of Arrhenius process rate matrix.
    """
    class Symbols:
        @classmethod
        def _barrier_element_symbol(cls, i, j):
            if i == j:
                return 0
            return sp.symbols('B_%d%d' % (i + 1, j + 1), real=True)

        def __init__(self, N):
            self.E_i = sp.symbols('E_1:%d' % (N + 1), real=True)
            self.B_ij = sp.Matrix(N, N, self._barrier_element_symbol)
            self.T = sp.symbols('T', real=True)

    @classmethod
    def _create_elements(cls, N):
        symbols = cls.Symbols(N)

        def create_symbolic_rate_matrix_element(i, j):
            if i == j:
                return 0
            return sp.exp(- (symbols.B_ij[i, j] - symbols.E_i[j]) / symbols.T)

        rate_matrix_symbolic = sp.Matrix(N, N, create_symbolic_rate_matrix_element)

        # Set each diagonal element as minus the sum of the other elements in its column (ensures Detailed Balance)
        rate_matrix_symbolic -= sp.diag(*np.sum(rate_matrix_symbolic, axis=0))

        return rate_matrix_symbolic, symbols

    def __new__(cls, N):
        """
        Parameters
        ----------
        N : int
            Number of states.
        """
        elements, symbols = cls._create_elements(N)
        self = super().__new__(cls, elements)
        self.symbols = symbols
        return self

    def subs_symbols(self, energies=None, barriers=None, temperature=None):
        """
        Return a new rate matrix with subs applied to each entry.

        Parameters
        ----------
        energies : 1-D array or sequence of float
            Energies of the states of the arrhenius, ordered in ascending order.
        barriers : 2-D array
            Matrix of energy barriers between states.
        temperature : float
            Temperature.

        Returns
        -------
        new : SymbolicRateMatrixArrhenius
            New instance of RateMatrixArrhenius with subs applied.
        """
        subs_dict = {}
        if energies is not None:
            subs_dict.update(zip(self.symbols.E_i, energies))
        if barriers is not None:
            subs_dict.update(zip(np.ravel(self.symbols.B_ij), np.ravel(barriers)))
            del subs_dict[0]
        if temperature is not None:
            subs_dict.update({self.symbols.T: temperature})
        expr = self.subs(subs_dict)
        if not expr.free_symbols:
            expr = np.array(expr).astype(np.float64)
        return expr

    def lambdify(self, symmetric_barriers=False):
        params = (self.symbols.T,) + self.symbols.E_i

        if symmetric_barriers:
            barriers_subs = dict(zip(np.ravel(np.triu(self.symbols.B_ij.T)),
                                     np.ravel(np.triu(self.symbols.B_ij))))

            barriers_free_symbols = set(barriers_subs.values())
            expr = self.subs(barriers_subs)
        else:
            barriers_free_symbols = set(self.symbols.B_ij.values())
            expr = self

        params += tuple(filter(lambda b: b in barriers_free_symbols, self.symbols.B_ij.values()))

        return sp.lambdify(params, expr)


class _SymbolicThreeStateEigensystem:
    FILE_NAME_EIGENSYSTEM = 'three_state_eigensystem_symbolic.pickle'

    @classmethod
    def _file_path(cls):
        import os
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        return os.path.join(__location__, cls.FILE_NAME_EIGENSYSTEM)

    @classmethod
    def _save_eigensystem(cls):
        r_sym = SymbolicRateMatrixArrhenius(3)
        eigensystem_right = r_sym.eigenvects()
        eigensystem_left = r_sym.T.eigenvects()

        eigenvalues, _, V = zip(*eigensystem_right)
        _, _, U = zip(*eigensystem_left)

        # The returned eigenvalues (from sympy) is ordered as: lam1, lam3, lam2 (seen in numerical checks)
        u1, u3, u2 = [sp.Matrix(U[i][0]) for i in [0, 1, 2]]
        lam1, lam3, lam2 = eigenvalues
        v1, v3, v2 = [sp.Matrix(V[i][0]) for i in [0, 1, 2]]

        # Normalization of left eigenvectors by their sum of their components
        u1 = sp.simplify(u1 / (np.sum(u1) / 3.))
        u2 = u2 / (np.sum(u2) / 3.)
        u3 = u3 / (np.sum(u3) / 3.)

        # Normalization of right eigenvectors by the inner product with the left eigenvectors
        v1 = v1 / u1.dot(v1)
        v2 = v2 / u2.dot(v2)
        v3 = v3 / u3.dot(v3)

        es = (u1, u2, u3), (lam1, lam2, lam3), (v1, v2, v3)

        pickle.dump(es, open(cls._file_path(), 'wb'))

    @classmethod
    def load_eigensystem(cls):
        return pickle.load(open(cls._file_path(), 'rb'))


def symbolic_three_state_eigensystem():
    return _SymbolicThreeStateEigensystem.load_eigensystem()
