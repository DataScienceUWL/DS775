# execute this cell for a *slow* animation of the local search
from IPython.display import display, clear_output
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# load problem data
with open("data/Caps48.json", "r") as tsp_data:
    tsp = json.load(tsp_data)
dist_mat = tsp["DistanceMatrix"]
optimal_tour = tsp["OptTour"]
opt_dist = tsp["OptDistance"]/1000 # converted to kilometers
xy_meters = np.array(tsp["Coordinates"])

def sub_tour_reversal(tour):
    # reverse a random tour segment
    num_cities = len(tour)
    i, j = np.sort(np.random.choice(num_cities, 2, replace=False))
    return np.concatenate((tour[0:i], tour[j:-num_cities + i - 1:-1],
                              tour[j + 1:num_cities]))

def tour_distance(tour, dist_mat):
    distance = dist_mat[tour[-1]][tour[0]]
    for gene1, gene2 in zip(tour[0:-1], tour[1:]):
        distance += dist_mat[gene1][gene2]
    return distance/1000 # convert to kilometers

# initialize with a random tour
n = 48
current_tour = np.random.permutation(np.arange(n))
current_dist = tour_distance(current_tour, dist_mat)

# plot initial tour
meters_to_pxl = 0.0004374627441064968
intercept_x = 2.464
intercept_y = 1342.546
xy_pixels = np.zeros(xy_meters.shape)
xy_pixels[:,0] = meters_to_pxl * xy_meters[:,0] + intercept_x
xy_pixels[:,1] = -meters_to_pxl * xy_meters[:,1] + intercept_y

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
im = plt.imread('images/caps48.png')
implot = ax.imshow(im)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='both', which='both', length=0)

loop_tour = np.append(current_tour, current_tour[0])
lines, = ax.plot(xy_pixels[loop_tour, 0],
        xy_pixels[loop_tour, 1],
        c='b',
        linewidth=1,
        linestyle='-')
dst_label = plt.text(100, 1200, '{:d} km'.format(int(current_dist)))

# local search with graphing
max_moves_no_improve = 5000
num_moves_no_improve = 0
while( num_moves_no_improve < max_moves_no_improve):
    num_moves_no_improve += 1
    new_tour = sub_tour_reversal(current_tour)
    new_dist = tour_distance(new_tour, dist_mat)
    if new_dist < current_dist:
        current_tour = new_tour
        current_dist = new_dist
        num_moves_no_improve = 0
        
        loop_tour = np.append(current_tour, current_tour[0])
        lines.set_data(xy_pixels[loop_tour, 0], xy_pixels[loop_tour, 1])
        dst_label.set_text('{:d} km'.format(int(current_dist)))
        clear_output(wait=True)
        display(fig)