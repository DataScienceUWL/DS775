import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np

x = np.linspace(-100, 100, 401)     
y = np.linspace(-100, 100, 401)     
X, Y = np.meshgrid(x, y) 
Z = (X**2) + (Y**2) + 20

data = [
    go.Surface( x = X, y = Y, z = Z, colorscale = 'Jet',
        contours=go.surface.Contours(
            z=go.surface.contours.Z(
              show=True,
              usecolormap=True,
              highlightcolor="#42f462",
              project=dict(z=True)
            )
        )
    )
]

layout = go.Layout(title='Sphere',width=600,height=600)
fig = go.Figure(data=data, layout=layout)
iplot(fig)