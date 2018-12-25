import numpy as np
import matplotlib.pyplot as plt


__all__ = ['figure', 'plot', 'scatter', 'point', 'equilibrium_line', 'eigenvectors', 'legend', 'savefig', 'show',
           'latex']


class _KwargsStyle:
    @classmethod
    def get_style(cls, style, **kwargs):
        _kwargs = dict(_KwargsStyle.style_dict.get(style, {}))
        if kwargs:
            _kwargs.update(kwargs)
        return _kwargs

    style_dict = {
        'canvas': {
            'c': 'k',
            'linewidth': 0.5
        },
        'equilibrium_line': {
            'label': 'Equilibrium line',
            'color': 'k',
            'linestyle': 'dashed',
            'linewidth': 1.5,
            'zorder': 0
        },
        'trajectory': {
            'linewidth': 2
        },
        'point': {
            's': 50,
            'zorder': 3
        },
        'legend': {
            'fontsize': 16,
            'markerscale': 1,
            'loc': 2
        },
        'arrow': {
            's': '',
            'xycoords': 'data',
            'textcoords': 'data',
            'arrowprops': dict(arrowstyle='->', connectionstyle='arc3', color='xkcd:dark grey', lw=1, ls='--')
        },
        'arrow_label': {
            's': '',
            'ha': 'right',
            'va': 'center',
            'fontsize': 12,
            'clip_on': True
        },
        'savefig': {
            'bbox_inches': 'tight',
            'dpi': 600,
            'transparent': True
        }
    }


def _transform_points(points):
    """
    Return the 2D projection on the L1 norm plane.

    1. Rotate around z axis by: 180+45 degrees (``theta_z``)
    2. Translate in y axis by: 1/sqrt(2) (``displacement_y``)
    3. Rotate around x axis by: -arctan(1/displacement) (``=theta_x``)
    """
    theta_z = np.deg2rad(180 + 45)
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

    displacement_y = 1. / np.sqrt(2)
    T_y = np.array([[0], [displacement_y], [0]])

    theta_x = -np.arctan(1. / displacement_y)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

    points = np.array(points)
    return R_x.dot(R_z.dot(points.reshape(3, -1)) + T_y)[:-1]


def extend_kwargs(style):
    """
    Extends kwargs for the decorated function by style.

    Parameters
    ----------
    style : str
        Style to include in kwargs.

    Returns
    -------
    deco : function
        Decorator that extends kwargs by the appropriate style.
    """
    def deco(plot_func):
        def wrapper(*args, **kwargs):
            plot_func(*args, **_KwargsStyle.get_style(style, **kwargs))
        return wrapper
    return deco


def latex(on=True):
    """
    Configure a latex figure style.

    Parameters
    ----------
    on : boolean, optional
        If True, use latex rendering.
        If False, do not use latex rendering.
    """
    plt.rcdefaults()
    if on:
        plt.rc('text', usetex=True)
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['font.size'] = 12
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


def plot(trajectories, **kwargs):
    """
    Plot trajectories in the probability plane.

    Parameters
    ----------
    trajectories : (3, K, M) or (3, M) array
        Trajectories to draw. ``K`` is the number of continuous curves, and ``M`` in the length of each curve.

    kwargs : Line2D properties, optional
        Used for `matplotlib.pyplot.plot` function (see documentation of this function for more details).

    Returns
    -------
    lines : list of Line2D objects
        A list of Line2D objects representing the plotted data.
    """
    if trajectories.ndim == 2:
        trajectories = np.expand_dims(trajectories, axis=1)

    trajectory_list = np.split(trajectories, trajectories.shape[1], axis=1)

    # Set the label keyword argument for only one trajectory (for legend)
    res = plt.plot(*_transform_points(trajectory_list[0]), **kwargs)
    if 'label' in kwargs:
        del kwargs['label']

    for trajectory in trajectory_list[1:]:
        res += plt.plot(*_transform_points(trajectory), **kwargs)

    return res


def scatter(points, **kwargs):
    """
    A scatter plot of points in the probability plane.

    Parameters
    ----------
    points : (3, K, M) array
        Points to draw. ``K`` is the number of curves and ``M`` in the number of points in each curve.
    kwargs : Line2D properties, optional
        Used for `matplotlib.pyplot.scatter` function (see documentation of this function for more details).

    Returns
    -------
    paths : PathCollection
    """
    return plt.scatter(*_transform_points(points), **kwargs)


@extend_kwargs('point')
def point(p, **kwargs):
    """

    Parameters
    ----------
    p : (3, 1) array
        Probability distribution point to draw
    kwargs : Line2D properties, optional
        Used for `matplotlib.pyplot.scatter` function (see documentation of this function for more details).

    Returns
    -------
    paths : PathCollection
    """
    return scatter(p, **kwargs)


def text(p, s, delta_x=0, delta_y=0, **kwargs):
    """
    Draw text.

    Parameters
    ----------
    p : (3, 1) array
        Point at which to locate the text (N=3).
    s : str
        Text to draw
    delta_x : float
        Displacement from ``p`` in the x axis.
    delta_y : float
        Displacement from ``p`` in the y axis.
    kwargs : Additional kwargs
        Used for `matplotlib.pyplot.text` function (see documentation of this function for more details).

    Returns
    -------
    text : Text
        The created Text instance.
    """
    x, y = _transform_points(p)
    return plt.text(x + delta_x, y + delta_y, s, kwargs)


def figure(fill=True, focus=False, ticket=False, **kwargs):
    """
    Create a new figure of the probability plane.

    Parameters
    ----------
    fill : boolean, optional, default: True
        Fill the probability plane with light yellow color.
    focus : boolean, optional, default: False
        Show lower left of the probability plane.
    ticket : boolean, optional, default: True
        Show a text ticket for the probability plane.
    kwargs : Line2D properties, optional
        Used for `matplotlib.pyplot.plot` function (see documentation of this function for more details).
    """
    plt.figure(**kwargs)

    # Plot the probability plane (triangular plane)
    plot(np.eye(3)[:, [0, 1, 1, 2, 2, 0]], **_KwargsStyle.get_style('canvas'))

    if fill:
        plt.fill(*_transform_points(np.eye(3)), c='xkcd:yellow', alpha=0.5, zorder=0)

    if ticket:
        plt.text(-0.61, 0.2, 'Probability plane', fontsize=12, rotation=64, va='center', ha='center')

    if focus:
        plt.xlim([-0.73, 0.13])
        plt.ylim([-0.05, 0.5])

    # Clean axes ticks
    plt.xticks([])
    plt.yticks([])


@extend_kwargs('equilibrium_line')
def equilibrium_line(eq_line, **kwargs):
    """
    Plot the equilibrium line.

    Parameters
    ----------
    eq_line : (3, M) array
        Equilibrium line.
    kwargs : Line2D properties, optional
        Used for `matplotlib.pyplot.plot` function (see documentation of this function for more details).
    """
    plot(eq_line, **kwargs)


@extend_kwargs('legend')
def legend(**kwargs):
    plt.legend(**kwargs)


@extend_kwargs('savefig')
def savefig(fname, **kwargs):
    plt.savefig(fname, **kwargs)


show = plt.show


def _plot_arrow_2d(point_from, point_to, kwargs_arrow=None, kwargs_label=None):

    # Extends annotate function kwargs with style 'arrow' and the given kwargs_arrow
    _kwargs_arrow = {} if kwargs_arrow is None else kwargs_arrow
    _kwargs_arrow = _KwargsStyle.get_style('arrow', xy=point_to.ravel(), xytext=point_from.ravel(), **_kwargs_arrow)
    plt.annotate(**_kwargs_arrow)

    # Extends text function kwargs with style 'arrow_label' and the given kwargs_label
    _kwargs_label = {} if kwargs_label is None else kwargs_label
    _kwargs_label = _KwargsStyle.get_style('arrow_label', **_kwargs_label)
    plt.text(*(point_to + np.array([[0.01], [0.015]])).ravel(), **_kwargs_label)


def eigenvectors(eigensystem, kwargs_arrow=None, kwargs_label=None):
    """
    Draw the directions of the second and third right eigenvectors.

    Draw two arrows from the equilibrium point in the directions of the second and third eigenvectors.

    Parameters
    ----------
    eigensystem : ((N, N), (N,), (N, N) array
        The eigensystem from which to draw, as a tuple of the left eigenvectors, the eigenvalues and the right
        eigenvectors (N=3). The same as the return of `markoviandynamics.numeric.rate_matrix.eigensystem`.
    kwargs_arrow : Additional kwargs
        Used for `matplotlib.pyplot.annotate` function (see documentation of this function for more details).
    kwargs_label : Additional kwargs
        Used for `matplotlib.pyplot.text` function (see documentation of this function for more details).
    """
    p_eq, v2, v3 = np.hsplit(eigensystem[2], 3)
    p_eq_2d = _transform_points(p_eq)

    # Add a vector to p_eq vector, transform to 2D and normalize to the given length
    def v_from_eq_normalized(v, length=1.0):
        p_eq_plus_v_2d = _transform_points(p_eq + v)
        v_2d = p_eq_plus_v_2d - p_eq_2d
        v_2d_normalized = v_2d / np.linalg.norm(v_2d)
        return v_2d_normalized * length

    v2_2d_normalized = v_from_eq_normalized(v2, 0.1)
    v3_2d_normalized = v_from_eq_normalized(v3, 0.1)

    _plot_arrow_2d(p_eq_2d, p_eq_2d + v2_2d_normalized,
                   kwargs_arrow=kwargs_arrow, kwargs_label={'s': r'$\left|v_{2}\right>$', **(kwargs_label or {})})
    _plot_arrow_2d(p_eq_2d, p_eq_2d + v3_2d_normalized,
                   kwargs_arrow=kwargs_arrow, kwargs_label={'s': r'$\left|v_{3}\right>$', **(kwargs_label or {})})


def trajectory_arrows(trajectory, at_indices, color='k', head_size=1.):
    """
    Draw arrows on a trajectory to emphasise its direction.

    Parameters
    ----------
    trajectory : (N, 1, K) array
        Trajectory on which to draw the arrows (N=3).
    at_indices : array or sequence
        The indices of the trajectory, from 0 to K-1, at which to draw an arrow.
    color : color
        Color of the arrow.
    head_size : float
        Size of the arrow, in arbitrary units.
    """
    trajectory_2d = _transform_points(trajectory)
    hw = 0.02 * head_size
    if np.size(at_indices) == 1:
        at_indices = [at_indices]
    for at_index in at_indices:
        x1, y1 = trajectory_2d[:, at_index]
        x2, y2 = trajectory_2d[:, at_index+1]
        dx, dy = x2 - x1, y2 - y1
        plt.arrow(x1, y1, dx, dy, head_width=hw, head_length=hw, lw=0, color=color)
