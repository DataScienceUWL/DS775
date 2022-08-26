import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np

def f(xy):
    obj = 0.2 + sum(xy**2 - 0.1*np.cos(6*np.pi*xy))
    return obj

numx = 101
numy = 101
x = np.linspace(-1.0, 1.0, numx)
y = np.linspace(-1.0, 1.0, numy)
xx,yy=np.meshgrid(x,y)
zz = np.zeros((numx,numy))
for i in range(numx):
    for j in range(numy):
        zz[i,j] = f(np.array([xx[i,j],yy[i,j]]))

data = [
    go.Surface( x = xx, y = yy, z = zz, colorscale = 'Jet')
]

layout = go.Layout(title='Bumpy',width=600,height=600)
fig = go.Figure(data=data, layout=layout)
iplot(fig)