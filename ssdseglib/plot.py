from matplotlib import get_backend

def move_figure(fig, x, y):
    """
    move matplotlib figure to x, y pixel on screen

    :param fig: matplotlib figure
    :param x: int, x location
    :param y: int, y location
    :return: nothing
    """

    # retrieve backend in use by matplotlib
    backend = get_backend()

    # move figure in the right place
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))

    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((x, y))

    else:
        # this works for qt and gtk
        fig.canvas.manager.window.move(x, y)
