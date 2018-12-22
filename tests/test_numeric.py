import pytest
import numpy as np
from markoviandynamics.numeric.dynamics import evolve
from markoviandynamics.numeric.rate_matrix import eigensystem, random_rate_matrix_arrhenius, decompose
from numpy.testing import assert_almost_equal


@pytest.fixture(params=[3, 5])
def N(request):
    return request.param


@pytest.fixture
def p_initial(N):
    p = np.random.rand(N, 1)
    p = p / np.sum(p, axis=0)
    return p


@pytest.fixture
def p_initials(N):
    p = np.random.rand(N, 5)
    p = p / np.sum(p, axis=0)
    return p


@pytest.fixture
def p_initials_3dim(N):
    p = np.random.rand(N, 5, 10)
    p = p / np.sum(p, axis=0)
    return p


@pytest.fixture
def rate_matrix(N):
    return random_rate_matrix_arrhenius(N)


@pytest.fixture
def rate_matrix_time(N):
    mat = np.stack([random_rate_matrix_arrhenius(N) for _ in range(10)], axis=2)
    return mat


class TestEvolution:

    @pytest.fixture
    def t_range(self):
        return np.arange(5)

    @pytest.fixture
    def p_time(self, p_initial, rate_matrix_time, t_range):
        return evolve(p_initial, rate_matrix_time, t_range)

    @pytest.fixture
    def p_time_multiple(self, p_initials, rate_matrix_time, t_range):
        return evolve(p_initials, rate_matrix_time, t_range)

    def test_evolve(self, p_initial, rate_matrix_time, t_range):
        p_time = evolve(p_initial, rate_matrix_time, t_range)
        assert np.shape(p_time) == p_initial.shape + (np.size(t_range),)

    def test_evolve_multiple_p(self, p_initials, rate_matrix_time, t_range):
        p_next = evolve(p_initials, rate_matrix_time, t_range)
        assert np.shape(p_next) == p_initials.shape + (np.size(t_range),)


class TestRateMatrixEigensystem:

    @pytest.fixture
    def es(self, rate_matrix):
        return eigensystem(rate_matrix)

    def test_eigensystem(self, rate_matrix, es):
        U, eigenvalues, V = es
        assert_almost_equal(U[:,0], 1)
        assert_almost_equal(U.T @ V, np.eye(np.size(eigenvalues)))
        assert_almost_equal(rate_matrix.dot(V), eigenvalues * V)
        assert_almost_equal(U.T.dot(rate_matrix), (U*eigenvalues).T)
        assert_almost_equal(V.dot(np.diag(eigenvalues)).dot(U.T), rate_matrix)

    def test_eigensystem_nd(self, rate_matrix_time):
        U, eigenvalues, V = eigensystem(rate_matrix_time)

        for i in range(rate_matrix_time.shape[2]):

            # Normalization - first left eigenvector elements are 1
            assert_almost_equal(U[:,0,i], 1)

            # Normalization - first right eigenvector (equilibrium) elements summed to 1
            assert_almost_equal(np.sum(V[:,0,i], axis=0), 1)

            # Normalization - the rest of the right eigenvector elements summed to 1
            assert_almost_equal(np.sum(V[:,1:V.shape[0],i], axis=0), 0)

            # Orthogonality
            assert_almost_equal(U[:,:,i].T @ V[:,:,i], np.eye(rate_matrix_time.shape[0]))

            # Right eigenvectors
            assert_almost_equal(rate_matrix_time[:,:,i] @ V[:,:,i], eigenvalues[:,:,i] * V[:,:,i])

            # Left eigenvectors
            assert_almost_equal(U[:,:,i].T @ rate_matrix_time[:,:,i], (U[:,:,i] * eigenvalues[:,:,i]).T)


def test_decompose(p_initial, p_initials, p_initials_3dim, rate_matrix, rate_matrix_time):
    U, eigenvalues, V = eigensystem(rate_matrix)
    assert_almost_equal(V @ decompose(p_initial, rate_matrix), p_initial)
    assert_almost_equal(V @ decompose(p_initials, rate_matrix), p_initials)

    U, eigenvalues, V = eigensystem(rate_matrix_time)

    decomposition = decompose(p_initial, rate_matrix_time)
    for i in range(rate_matrix_time.shape[2]):
        assert_almost_equal(V[:,:,i] @ decomposition[:,:,i], p_initial)

    decomposition = decompose(p_initials, rate_matrix_time)
    for i in range(rate_matrix_time.shape[2]):
        assert_almost_equal(V[:,:,i] @ decomposition[:,:,i], p_initials)

    decomposition = decompose(p_initials_3dim, rate_matrix_time)
    for i in range(rate_matrix_time.shape[2]):
        assert_almost_equal(V[:,:,i] @ decomposition[:,:,i], p_initials_3dim[:,:,i])
