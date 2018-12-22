import numpy as np


def temperature_array_from_segments(segment_temperatures, segment_start_times, t_range):
    """
    Compute temperature array in time from a temperature segments and their start time.

    The returned temperature array is a piecewise constant discrete function, where each piece is a segment described
    by the its temperature and the time it starts.

    Parameters
    ----------
    segment_temperatures : (S,) array
        Temperatures sequence represent the segment temperatures.
    segment_start_times : (S,) array
        Start time sequence represent the segment start time.
    t_range :(K,) array
        Time array.

    Returns
    -------
    temperature_array : (K,) array
        Temperature array.
    """
    temperature_array = np.zeros_like(t_range)
    for step, (from_, to) in enumerate(zip(segment_start_times[:-1], segment_start_times[1:])):
        temperature_array[np.logical_and(from_ <= t_range, t_range < to)] = segment_temperatures[step]

    # Fill the rest of the array to the last segment temperature
    temperature_array[segment_start_times[-1] <= t_range] = segment_temperatures[-1]
    return temperature_array


def distance(p1, p2):
    """
    Distance between two probability distributions, using Kullback-Leibler divergence.

    Parameters
    ----------
    p1 : (N, M) or (N, K, M) array
        Probability distribution.
    p2 : (N, M) or (N, K, M) array
        Probability distribution against which the the distance is computed.

    Returns
    -------
        d : (M,) or (K, M) array
            Distance between ``p1`` and ``p2``.
    """
    return np.sum((- p1.T * np.log(p2.T) + p1.T * np.log(p1.T)).T, axis=0)
