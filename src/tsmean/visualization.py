from tsmean.warping_distance import dtw
import numpy as np
import matplotlib.pyplot as plt

from enum import IntEnum


class LegendStyle(IntEnum):
    """
    Enum representing different legend styles for mean alignment plots.

    NONE : No legend
    STD : Standard legend
    LEFT_AT_TIME_SERIES : Legend at the left of the time series
    RIGHT_AT_TIME_SERIES : Legend at the right of the time series
    """
    
    NONE = 0
    STD = 1
    LEFT_AT_TIME_SERIES = 2
    RIGHT_AT_TIME_SERIES = 3




def alignment_plot(x1, x2, path=None, *, 
                    # Plot configuration
                    figsize=(6.4, 4.8), offset=True,
                    # Styling - Series x1
                    x1_color='k', x1_line_width=1,
                    # Styling - Series x2
                    x2_color='k', x2_line_width=1,
                    # Styling - Paths
                    p_color='gray', p_line_width=0.5,
                    # Labels & Legend
                    title=None, xlabel=None, ylabel=None,
                    legend_style=LegendStyle.RIGHT_AT_TIME_SERIES, latex=True, font_size=12,
                    # Return
                    return_fig=False):
    """
    Plot the alignment of two time series.
    
    Parameters
    ----------
    x1 : array_like
        First time series.
    x2 : array_like
        Second time series.
    path : list of 2-tuples of ints (i,j), optional
        Warping path. If None, the alignment is computed using dtw.
    
    figsize : tuple, optional
        Figure size.
    offset : bool, optional
        Whether to vertically offset the first time series.
        
    x1_color : str, optional
        Color of the first time series.
    x1_line_width : float, optional
        Line width of the first time series.
        
    x2_color : str, optional
        Color of the second time series.
    x2_line_width : float, optional
        Line width of the second time series.
        
    p_color : str, optional
        Color of the alignment.
    p_line_width : float, optional
        Line width of the alignment.
        
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        X label of the plot.
    ylabel : str, optional
        Y label of the plot.
    legend_style : LegendStyle, optional
        Legend style.
    latex : bool, optional
        Whether to use LaTeX for labels.
    font_size : int, optional
        Font size.
        
    return_fig : bool, optional
        Whether to return the figure.
    """
    min1 = min(x1)
    max2 = max(x2)
    off = (max2 - min1 + 1) if offset else 0

    plt.figure(figsize=figsize)

    # plot alignment
    for i, j in path:
        plt.plot([i, j], [x1[i] + off, x2[j]], color=p_color, linewidth=p_line_width)

    # disable y axis if offset
    if offset:
        plt.gca().get_yaxis().set_visible(False)

    # define text labels
    x1_text = r'$x_1$' if latex else 'x1'
    x2_text = r'$x_2$' if latex else 'x2'
    
    # calculate max text width
    h_offset = __get_max_text_width(plt, [x1_text,x2_text], font_size)
    max_len = max(len(x1), len(x2))
    h_shift = max_len/100

    plt.plot(x1 + off, color=x1_color, label=x1_text, linewidth=x1_line_width)
    plt.plot(x2, color=x2_color, label=x2_text, linewidth=x2_line_width)
    if legend_style == LegendStyle.STD:
        plt.legend()
    elif legend_style == LegendStyle.LEFT_AT_TIME_SERIES:
        plt.text(-h_shift, x1[0] + off, x1_text, ha='right', va='center', fontsize=font_size)
        plt.text(-h_shift, x2[0] , x2_text, ha='right', va='center', fontsize=font_size)
        plt.xlim(-(h_offset+3*h_shift), max_len + h_shift)

    elif legend_style == LegendStyle.RIGHT_AT_TIME_SERIES:
        plt.text(len(x1)-1+h_shift, x1[-1] + off, x1_text, ha='left', va='center', fontsize=font_size)
        plt.text(len(x2)-1+h_shift, x2[-1] , x2_text, ha='left', va='center', fontsize=font_size)
        plt.xlim(-h_shift, max_len -1 + (h_offset+2*h_shift))

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.tick_params(axis='both', labelsize=font_size)
    
    if return_fig:
        fig = plt.gcf()
        plt.close(fig)
        return fig
    else:
        plt.show()



def mean_alignment_plot(X, mu, paths=None, *,
                    # Plot configuration
                    figsize=(6.4, 4.8), offset=True,
                    arrangement='auto', h_stretch=1.1, v_stretch=0.6,
                    # Styling - Series X
                    x_colors=None, x_line_width=1, x_texts=None,
                    # Styling - Mean mu
                    mu_color='blue', mu_line_width=2, mu_text=None,
                    # Styling - Paths
                    p_colors=None, p_line_width=0.5, p_alpha=1,
                    # Labels & Legend
                    title=None, xlabel=None, ylabel=None, 
                    show_x_axis=False, legend_style=LegendStyle.STD, latex=True, font_size=10,
                    # Return
                    return_fig=False):
    """
    Plots multiple time series aligned to their mean (barycenter).
    
    The time series are arranged around the mean to visualize the warping.
    
    Parameters
    ----------
    X : list of np.ndarray
        List of time series to plot.
    mu : np.ndarray
        The mean time series (barycenter).
    paths : list of list of tuples, optional
        Warping paths for each time series in X to mu. 
        If None, paths are computed using basic DTW.
    
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (6.4, 4.8).
    offset : bool, optional
        If True, spreads time series out. If False, plots everything on top of each other.
    arrangement : str or list of tuples, optional
        Arrangement strategy for the time series around the mean.
        'auto': Automatically arranges series.
        list of tuples: Manual positions [(h_pos, v_pos), ...].
    h_stretch : float, optional
        Horizontal spacing factor for arrangement.
    v_stretch : float, optional
        Vertical spacing factor for arrangement.
        
    x_colors : list of colors or str, optional
        Colors for the time series in X. If None, automatic colors.
    x_line_width : float, optional
        Line width for time series in X.
    x_texts : list of str, optional
        Labels for the time series in X.
        
    mu_color : str, optional
        Color for the mean mu.
    mu_line_width : float, optional
        Line width for the mean mu.
    mu_text : str, optional
        Label for the mean mu.
        
    p_colors : list of colors or str, optional
        Colors for the alignment paths. If None, inherits from x_colors.
    p_line_width : float, optional
        Line width for alignment paths.
    p_alpha : float, optional
        Opacity (alpha) for alignment paths.
        
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    show_x_axis : bool, optional
        If True, shows the x-axis ticks and labels.
    legend_style : LegendStyle, optional
        Style of the legend (NONE, STD, LEFT_AT_TIME_SERIES, RIGHT_AT_TIME_SERIES).
    latex : bool, optional
        If True, uses LaTeX formatting for default labels.
    font_size : int, optional
        Font size for labels and text.
        
    return_fig : bool, optional
        If True, returns the Figure object instead of showing it.
    
    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        The figure object if return_fig is True.
    """
    n_series = len(X)

    if paths is None:
        paths = [dtw(mu,x)[1] for x in X]
    
    # Automatic colour selection if not specified
    if x_colors is None:
        if n_series <= 10:
            cmap = plt.get_cmap('tab10')
            x_colors = [cmap(i) for i in range(n_series)]
        elif n_series <= 20:
            cmap = plt.get_cmap('tab20')
            x_colors = [cmap(i) for i in range(n_series)]
        else:
            # For an arbitrary number of time series: continuous Colormap
            cmap = plt.get_cmap('hsv')  # or 'rainbow', 'viridis', 'plasma'
        x_colors = [cmap(i / n_series) for i in range(n_series)]
    elif isinstance(x_colors, str):
        x_colors = [x_colors] * n_series

    if p_colors is None:
        p_cols = x_colors
    elif isinstance(p_colors, str):
        p_cols = [p_colors] * n_series
    elif isinstance(p_colors, list):
        p_cols = p_colors
    else:
        raise ValueError("p_color must be a color or a list of colors")

    # Automatic arrangement of the time series
    if arrangement == 'auto':
        # Standard positions for different n
        if n_series == 1:
            positions = [(0, 1)]  # top
        elif n_series == 2:
            positions = [(0, 1), (0, -1)]  # top, bottom
        elif n_series == 3:
            positions = [(-1, 1), (1, 1), (0, -1)]  # left-top, right-top, middle-bottom
        elif n_series == 4:
            positions = [(-1, 1), (1, 1), (-1, -1), (1, -1)]  # 2x2 Grid
        elif n_series == 5:
            positions = [(-1, 1), (1, 1), (0, 0.5), (-1, -1), (1, -1)]
        elif n_series == 6:
            positions = [(-1, 1), (0, 1), (1, 1), (-1, -1), (0, -1), (1, -1)]
        else:
            # For more time series: spiral arrangement
            positions = []
            angles = np.linspace(0, 2*np.pi, n_series, endpoint=False)
            for angle in angles:
                h_pos = np.cos(angle)
                v_pos = np.sin(angle)
                positions.append((h_pos, v_pos))
    else:
        positions = arrangement
    
    # Calculate Offsets
    h_offsets = np.zeros(n_series)  # horizontal Offsets
    v_offsets = np.zeros(n_series)  # vertical Offsets
    
    if offset:
        # Calculate distances
        max_len = max(len(x) for x in X)
        h_spacing = max_len * h_stretch  # h_stretch of the length as horizontal distance
        
        ranges = [max(x) - min(x) for x in X]
        max_range = max(max(mu) - min(mu), max(ranges))
        v_spacing = max_range * v_stretch  # v_stretch of the range as vertical distance
        
        for idx, (h_pos, v_pos) in enumerate(positions):
            h_offsets[idx] = h_pos * h_spacing
            
            # Vertical Offset relative to the mean
            if v_pos > 0:  # above
                v_offsets[idx] = max(mu) - min(X[idx]) + v_spacing * abs(v_pos)
            elif v_pos < 0:  # below
                v_offsets[idx] = min(mu) - max(X[idx]) - v_spacing * abs(v_pos)
            else:  # same height as mean
                v_offsets[idx] = np.mean(mu) - np.mean(X[idx])
    
    # Plotting
    plt.figure(figsize=figsize)
    
    # Draw alignment lines
    for idx, (x, path) in enumerate(zip(X, paths)):
        for i, j in path:
            plt.plot([i, j + h_offsets[idx]], 
                    [mu[i], x[j] + v_offsets[idx]], 
                    color=p_cols[idx], linewidth=p_line_width, alpha=p_alpha)
    
    if offset:
        plt.gca().get_yaxis().set_visible(False)
    
    if not show_x_axis:
        plt.gca().get_xaxis().set_visible(False)
    
    # Prepare text labels
    if x_texts is None:
        if latex:
            x_texts = [f'$x_{{{i+1}}}$' for i in range(n_series)]
        else:
            x_texts = [f'x{i+1}' for i in range(n_series)]
    if mu_text is None:
        if latex:
            mu_text = r'$\mu$'
        else:
            mu_text = 'mean'

    # Calculate Text-Offset
    h_offset = __get_max_text_width(plt, x_texts + [mu_text], font_size)
    max_len = max(max(len(x) for x in X), len(mu))
    h_shift = max_len / 50
    
    # Plot the time series
    for idx, x in enumerate(X):
        x_indices = np.arange(len(x)) + h_offsets[idx]
        plt.plot(x_indices, x + v_offsets[idx], 
                color=x_colors[idx], label=x_texts[idx], 
                linewidth=x_line_width)
    
    # Plot the mean
    plt.plot(mu, color=mu_color, linewidth=mu_line_width, label=mu_text)
    
    # Legend/Labels
    if legend_style == LegendStyle.STD:
        plt.legend()
    elif legend_style == LegendStyle.LEFT_AT_TIME_SERIES:
        for idx, x in enumerate(X):
            plt.text(h_offsets[idx] - h_shift, x[0] + v_offsets[idx], 
                    x_texts[idx], ha='right', va='center', 
                    fontsize=font_size, color=x_colors[idx])
        plt.text(-h_shift, mu[0], mu_text, ha='right', va='center', 
                fontsize=font_size, color=mu_color)
        xmin, xmax = plt.xlim()
        plt.xlim(xmin - h_offset, xmax)
        
    elif legend_style == LegendStyle.RIGHT_AT_TIME_SERIES:
        for idx, x in enumerate(X):
            plt.text(len(x)-1+h_shift+h_offsets[idx], x[-1] + v_offsets[idx], 
                    x_texts[idx], ha='left', va='center', 
                    fontsize=font_size, color=x_colors[idx])
        plt.text(len(mu)-1+h_shift, mu[-1], mu_text, ha='left', va='center', 
                fontsize=font_size, color=mu_color)
        xmin, xmax = plt.xlim()
        plt.xlim(xmin, xmax + h_offset)
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    plt.tick_params(axis='both', labelsize=font_size)
    
    if return_fig:
        fig = plt.gcf()
        plt.close(fig)
        return fig
    else:
        plt.show()


def __get_max_text_width(plt, texts, font_size):
    """
    Calculate the maximum width of the given texts in data coordinates.
    
    Helper function for alignment_plot and mean_alignment_plot. Useful for 
    adjusting the xlim of the plot to make room for the legend.

    Parameters
    ----------
    plt : matplotlib.pyplot
        The plot.
    texts : list of str
        The texts.
    font_size : int
        The font size.
    
    Returns
    -------
    float
        The maximum width of the texts in data coordinates.
    """
    max_width = 0
    for text in texts:
        # 1. Create the text object
        t = plt.text(0, 0, text, fontsize=font_size)
        # 2. Force a draw so the renderer calculates the size
        plt.gcf().canvas.draw() 
        # 3. Get the bounding box in display coordinates (pixels)
        bbox = t.get_window_extent()
        width_pixels = bbox.width
        # 4. (Optional) Convert pixels to data coordinates
        # This is usually what you want if you are adjusting xlim based on text width
        inv = plt.gca().transData.inverted()
        bbox_data = bbox.transformed(inv)
        width_data = bbox_data.width
        t.remove()
        max_width = max(max_width, width_data)

    return max_width


