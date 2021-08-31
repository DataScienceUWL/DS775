import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# define objective function and show a contour plot

def f(xy):
    obj = 0.2 + sum(xy**2 - 0.1*np.cos(6*np.pi*xy))
    return obj

# compute on mesh for plotting
numx = 101
numy = 101
x = np.linspace(-1.0, 1.0, numx)
y = np.linspace(-1.0, 1.0, numy)
xx,yy=np.meshgrid(x,y)
z = np.zeros((numx,numy))
for i in range(numx):
    for j in range(numy):
        z[i,j] = f(np.array([xx[i,j],yy[i,j]]))

# Create a contour plot
plt.figure(figsize=(8,8))
# Plot contours
contours = plt.contour(xx,yy,z,30)
# Label contours
plt.clabel(contours, inline=1, fontsize=8)
# Add some text to the plot
plt.title('Non-Convex Function')
plt.xlabel('x')
plt.ylabel('y');