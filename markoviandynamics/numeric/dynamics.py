import numpy as np


__all__ = ['evolve']


def _evolve_single_time_step(p, rate_matrix, dt):
    return p + dt * rate_matrix @ p


def _evolution_iterator(p_initial, rate_matrix_time, t_range):
    p = np.copy(p_initial)
    num_of_steps = np.size(t_range)
    step_dt = np.diff(t_range)

    yield p
    for step in np.arange(num_of_steps - 1):
        rate_matrix = rate_matrix_time[:, :, step]
        p = _evolve_single_time_step(p, rate_matrix, step_dt[step])
        yield p


def _evolution_iterator_fixed_rate_matrix(p_initial, rate_matrix, t_range):
    p = np.copy(p_initial)
    num_of_steps = np.size(t_range)
    step_dt = np.diff(t_range)

    yield p
    for step in np.arange(num_of_steps - 1):
        p = _evolve_single_time_step(p, rate_matrix, step_dt[step])
        yield p


def evolve(p_initial, rate_matrix_time, t_range):
    """
    Evolves initial probability distribution in time.

    The initial probability distribution evolves in time using the given rate matrix. If ``rate_matrix_time`` is 2D, it
    will use throughout the whole evolution. If ``rate_matrix_time`` is 3D, it will be considered as stacked rate
    matrices in axis 2 where each matrix corresponds to each time step. Therefore, in this case ``len(t_range)`` must
    be equal to ``rate_matrix_time.shape[2]``.

    The array ``time_range`` can have varying time steps, namely ``np.diff(time_range)`` can have different values at
    each step. This way each step can be calculated using different dt(t), and can be useful in case of time range in
    log-scale.

    Parameters
    ----------
    p_initial : (N,) or (N, k) array
        Initial probability distributions, where k is the number of realizations.
    rate_matrix_time : (N, N) or (N, N, M) array
        Rate matrix or rate matrices stacked.
        If 2D, it will used as fixed for every time step.
        If 3D, every matrix will used in the corresponding time step.
    t_range : (M,) array_like
        Array of time points. Must start in zero and in ascending order.

    Returns
    -------
    res : (N, 1, len(t_range)) or (N, k, len(t_range)) array
        Trajectory - p(t) for every t in ``t_range``

    """
    if rate_matrix_time.ndim == 2 or rate_matrix_time.shape[2] == 1:
        iterator = _evolution_iterator_fixed_rate_matrix(p_initial, rate_matrix_time, t_range)
    else:
        iterator = _evolution_iterator(p_initial, rate_matrix_time, t_range)
    return np.stack(list(iterator), axis=2)
