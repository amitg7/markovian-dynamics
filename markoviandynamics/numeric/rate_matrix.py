import numpy as np
import sympy as sp
from markoviandynamics.symbolic import SymbolicRateMatrixArrhenius
from markoviandynamics.utils import temperature_array_from_segments


__all__ = ['rate_matrix_arrhenius', 'rate_matrix_arrhenius_time_segmented', 'eigensystem', 'decompose',
           'random_rate_matrix_arrhenius']


def _sort_eigensystem(eigenvectors_right, eigenvectors_left, eigenvalues_right, eigenvalues_left=None):
    """
    Sort by eigenvalues (descending). The left and right eigenvectors return in the corresponding order.

    Left eigenvectors are sorted by the left eigenvalues if given, otherwise by the right eigenvalues.
    """

    # Right
    _sorted_eigenvalues_indices = np.argsort(np.abs(eigenvalues_right))
    V = eigenvectors_right[:, _sorted_eigenvalues_indices]
    eigenvalues = -np.abs(eigenvalues_right[_sorted_eigenvalues_indices])

    # Left
    if eigenvalues_left is not None:
        _sorted_eigenvalues_indices = np.argsort(np.abs(eigenvalues_left))
    U = eigenvectors_left[:, _sorted_eigenvalues_indices]

    return U, eigenvalues, V


def _eigensystem_nd(a):
    a = np.moveaxis(a, 2, 0)

    eigenvalues, eigenvectors_right = np.linalg.eig(a)
    eigenvalues_left, eigenvectors_left = np.linalg.eig(np.swapaxes(a, 1, 2))

    evs = []
    V = []
    U = []

    for i in range(a.shape[0]):

        # Right
        _eigenvalues_right = eigenvalues[i, :]
        _eigenvectors_right = eigenvectors_right[i, :, :]

        # Left
        _eigenvalues_left = eigenvalues_left[i, :]
        _eigenvectors_left = eigenvectors_left[i, :, :]

        _U, _eigenvalues, _V = _sort_eigensystem(_eigenvectors_right, _eigenvectors_left,
                                                 _eigenvalues_right, _eigenvalues_left)

        # Normalization of left eigenvectors by their sum of their components (divided by 3 to get u1=(1, 1, 1))
        _U = _U / (np.expand_dims(_U.sum(axis=0), 0) / _U.shape[0])

        # Normalization of right eigenvectors by the inner product with the left eigenvectors
        _V = _V / np.diagonal(_U.T @ _V)

        _eigenvalues = np.expand_dims(_eigenvalues, 0)

        U.append(_U)
        evs.append(_eigenvalues)
        V.append(_V)

    return tuple(map(lambda x: np.stack(x, axis=2), [U, evs, V]))


def eigensystem(a):
    """
    Compute the eigenvalues, right and left eigenvectors of a square matrix or matrices.

    If the given ``a`` is a 3D array, this function computes the eigensystem for each square matrix along axis 2.
    For every square matrix, the eigenvalues returned in descending order (along axis 1).
    The left and right eigenvectors returned in the corresponding order.

    Parameters
    ----------
    a : (N, N) or (N, N, M) array
        Matrix or matrices for which the eigenvalues, right and left eigenvectors will be computed.

    Returns
    -------
    U : (N, N) or (N, N, M) array
        The left eigenvectors.
        If ``a`` is 2D: ``U[:,i]`` is the i'th left eigenvector.
        If ``a`` is 3D: ``U[:,i,j]`` is the i'th left eigenvector of the j'th matrix ``a[:,:,j]``.
    eigenvalues : (N,) or (N, M) array
        The eigenvalues.
        If ``a`` is 2D: ``eigenvalues[i]`` is the i'th eigenvalue.
        If ``a`` is 3D: ``eigenvalues[i]`` is the i'th eigenvalue of the 'j'th matrix ``a[:,:,j]``.
    V : (N, N)  or (N, N, M) array
    The right eigenvectors.
        If ``a`` is 2D: ``V[:,i]`` is the i'th right eigenvector.
        If ``a`` is 3D: ``V[:,i,j]`` is the i'th right eigenvector of the j'th matrix ``a[:,:,j]``.
    """
    if a.ndim == 3:
        return _eigensystem_nd(a)

    eigenvalues_right, eigenvectors_right = np.linalg.eig(a)
    eigenvalues_left, eigenvectors_left = np.linalg.eig(a.T)

    U, eigenvalues, V = _sort_eigensystem(eigenvectors_right, eigenvectors_left, eigenvalues_right, eigenvalues_left)

    # Normalization of left eigenvectors by their sum of their components (divided by 3 to get u1=(1, 1, 1))
    U = U / (U.sum(axis=0) / a.shape[0])

    # Normalization of right eigenvectors by the inner product with the left eigenvectors
    V = V / np.diagonal(U.T.dot(V))

    return U, eigenvalues, V


def rate_matrix_arrhenius(energies, barriers, temperature_array):
    """
    Compute the Arrhenius process rate matrix.

    For each temperature in ``temperature_array`` a rate matrix is computed using ``energies`` and ``barriers``
    and stacked in the depth dimension (axis=2).

    Parameters
    ----------
    energies : (N,) array or sequence of float
        Energies of the states of the arrhenius, ordered in ascending order.
    barriers : (N, N) array
        Energy barriers between states. Must be given as matrix.
    temperature_array : (M,) array or sequence of float
        Sequence of temperatures.

    Returns
    -------
    rate_matrix_time : (N, N, M)
        Rate matrices stacked.
    """
    N = np.size(energies)
    R = SymbolicRateMatrixArrhenius(N)

    rate_matrix_func = sp.lambdify(R.symbols.T, R.subs_symbols(energies, barriers))
    return rate_matrix_func(temperature_array)


def rate_matrix_arrhenius_time_segmented(energies, barriers, segment_temperatures, segment_start_times, t_range):
    """
    Compute the rate matrix for each time ``t`` in ``t_range``, where the bath temperature is a piecewise constant
    function of time.

    The bath temperature function, by which the rate matrices are calculated, is a piecewise constant function where
    each piece is a segment described by the its temperature and the time it starts.

    First, the temperature for every time, denoted by ``T(t)``, is calculated as follows:
    ``T(t) = Ti`` where ``t = segment_start_times[i]`` and ``Ti = segment_temperatures[i]``.
    Then, for every time ``t`` in ``t_range``, a rate matrix is calculated with the corresponding temperature ``T(t)``.

    The bath temperature is set to the last given temperature ``segment_start_times[-1]`` and stays at this value
    until the last time ``t`` in ``t_range``.

    Parameters
    ----------
    energies : (N,) array or sequence of float
        Energies of the states of the arrhenius, ordered in ascending order.
    barriers : (N, N) array
        Energy barriers between states. Must be given as matrix.
    segment_temperatures : (K,) array
        Temperature sequence where each temperature corresponds to each segment.
    segment_start_times : (K,) array
        Start time sequence where each time corresponds to each segment.
    t_range : (M,) array
        Time sequence.

    Returns
    -------
    rate_matrix_time : (N, N, M)
        Rate matrices stacked in the depth dimension (axis=2).

    Raises
    -----
    ValueError
        If the first segment start time ``segment_start_times[0]`` is not equal to ``t_range[0]``.
    """
    if segment_start_times[0] != t_range[0]:
        raise ValueError('The first segment start time `segment_start_times[0]` must be equal to `t_range[0]`.')

    temperature_array = temperature_array_from_segments(segment_temperatures, segment_start_times, t_range)
    return rate_matrix_arrhenius(energies, barriers, temperature_array)


def decompose(points, rate_matrix):
    """
    Compute the coefficients in ``points`` of the right eigenvectors of the rate matrix.

    The coefficients returned in the same order as the eigenvectors. Namely, ``decomposition[i, ...]`` are the
    coefficients of ``V[i, :]``.

    - If ``points`` and ``rate_matrix`` have the same number of dimensions, the result will have the same shape as
      ``points``. The third axis of both of them, if exists, must be the same size.

    - If ``points`` is 2-D and ``rate_matrix`` is 3-D, the results will be 3-D where the first two dimensions will be
      the same size as of ``points`` and the third dimension will be the same size as of ``rate_matrix``. In this case,
      the results are a stacked results (in the third dimension) of ``decompose(points, rate_matrix[:,:,i]`` for every
      index ``i`` in the third dimension of ``rate_matrix``.

    Parameters
    ----------
    points : (N, M) or (N, K, M) array
        Probability distributions to decompose.
    rate_matrix : (N, N) or (N, K, N) array
        Rate matrix of which to take the right eigenvectors.

    Returns
    -------
    decomposition : (N, M) or (N, K, M) array
        Decomposition of the given probability distributions.
    """
    U, _, _ = eigensystem(rate_matrix)

    if points.ndim == U.ndim == 2:
        return U.T.dot(points)

    if points.ndim == 3:
        points_iter = [points[:,:,i] for i in range(points.shape[2])]
        if U.ndim == 3:
            U_iter = [U[:,:,i] for i in range(U.shape[2])]
        else:
            U_iter = [U for _ in range(points.shape[2])]
    else:
        points_iter = [points for _ in range(U.shape[2])]
        U_iter = [U[:,:,i] for i in range(U.shape[2])]

    decomposition = []
    for p, U in zip(points_iter, U_iter):
        decomposition.append(U.T.dot(p))
    return np.stack(decomposition, axis=2)


def random_rate_matrix_arrhenius(N, max_barrier=2.5):
    """
    Compute random Arrhenius process rate matrix.

    Parameters
    ----------
    N : int
        Number of states.
    max_barrier : float
        Maximum energy for the random barriers.

    Returns
    -------
    res : (N, N) array
        Random rate matrix.
    """
    energies = np.sort(np.random.rand(N))
    barriers = np.zeros((N, N))

    # Iterate from the high energy state
    for j in range(N)[::-1]:

        # Compute random barrier to transition from (to) ``j`` state to (from) lower energy states
        for i in range(j):
            barriers[i, j] = barriers[j, i] = np.random.uniform(energies[i], max_barrier)

    return SymbolicRateMatrixArrhenius(N).subs_symbols(energies, barriers, temperature=1)
