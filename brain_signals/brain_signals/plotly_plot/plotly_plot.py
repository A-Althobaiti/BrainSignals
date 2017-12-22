from plotly.offline import plot
from scipy.stats.mstats import zscore
import plotly.graph_objs as go
import numpy as np

def enumerate_name(name, length):
    splitted_name = name.split()
    prefix = None
    
    if len(splitted_name) == 1:
        prefix = name[:2:]
    else:
        prefix = ''
        for single_name in splitted_name:
            prefix = prefix + single_name[0]

    return ['{}: {}'.format(prefix, index) for index in range(1, length + 1)]

class plotly_plot(object):
    
    data = None
    frames = None
    layout = None

    @classmethod
    def clean_plot(cls):
        cls.data = []
        cls.frames = []
        cls.layout = go.Layout()

    @classmethod
    def ggplot_style(cls):
        cls.layout.paper_bgcolor = 'rgb(243, 243, 243)'
        cls.layout.plot_bgcolor = 'rgb(229, 229, 229)'
        cls.layout.yaxis['gridcolor'] ='rgb(255, 255, 255)'
        cls.layout.yaxis['zerolinecolor'] = 'rgb(255, 255, 255)'
        cls.layout.xaxis['zerolinecolor'] = 'rgb(255, 255, 255)'

    @staticmethod
    def boxplot(y, name=None, **kwargs):
        plotly_plot.clean_plot()

        if len(y.shape) != 1:
            raise ValueError('y must be a one dimensional array')

        if name is None:
            name = 'Data Points'

        plotly_plot.layout.title = kwargs['title'] if 'title' in kwargs.keys() \
                                   else ''
        plotly_plot.layout.xaxis['range'] = [-2.5, 2.5]
        plotly_plot.layout.hovermode = 'closest'
        plotly_plot.ggplot_style()

        z_scores = zscore(y)

        scatter_trace = go.Scatter(x=np.random.uniform(-0.5, 0.5, len(y)),
                                    y=y,
                                    mode='markers',
                                    text=['<br>{}<br>z-score: {:.2f}'.format(s,z) for z, s \
                                        in zip(z_scores, enumerate_name(name, len(y)))],
                                    name=name)

        boxplot_trace = go.Box(y=y,
                               boxpoints=False,
                               boxmean=True,
                               name='Box Plot')

        plotly_plot.data = [scatter_trace, boxplot_trace]
        fig = go.Figure(data=plotly_plot.data,
                        frames=plotly_plot.frames,
                        layout=plotly_plot.layout)

        plot(fig)

    @staticmethod
    def line_chart(names=None, title=None, *lines):
        plotly_plot.clean_plot()

        if len(lines) == 0:
            raise ValueError('No line data points have been provided.')

        if names is None:
            names = ['Data Points'] * len(lines)

        elif isinstance(names, str) and len(lines) != 1:
            raise ValueError('names for all lines must be provided.')

        elif isinstance(names, list) and len(names) != len(lines):
            raise ValueError('the provided name(s) must have the same length'\
                            ' as lines.')

        plotly_plot.layout.title = title if title is not None else ''
        plotly_plot.ggplot_style()

        for line, name in zip(lines, names):
            line_plot = go.Scatter(
                x=np.linspace(1, len(line), len(line)),
                y=line,
                mode=line,
                name=name,
                hoverinfo='none',
                line={'width':1}
                )
            plotly_plot.data.append(line_plot)

        plotly_plot.layout.xaxis['title'] = 'Time'
        plotly_plot.layout.yaxis['title'] = 'Y'

        fig = go.Figure(data=plotly_plot.data,
                        frames=plotly_plot.frames,
                        layout=plotly_plot.layout)

        plot(fig)
