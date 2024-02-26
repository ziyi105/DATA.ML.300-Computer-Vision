from linefitlsq import linefitlsq
import numpy as np
import matplotlib.pyplot as plt

# Load and plot points
data = np.load('points.npy')
x, y = data[0, :], data[1, :]
plt.figure(1, (10, 10))
plt.plot(x, y, 'kx')
plt.axis('scaled')

# RANSAC parameters
# m is the number of data points
m = np.size(x) * 1.0
# s is the size of the random sample
s = 2
# t is the inlier distance threshold
t = np.sqrt(3.84) * 2
# e is the expected outlier ratio
e = 0.8
# at least one random sample should be free 
# from outliers with probability p
p = 0.999
# required number of samples
N_estimated = np.log(1 - p) / np.log(1 - (1 - e) ** s)

############### RANSAC loop ######################

# First initialize some variables
N = np.inf
sample_count = 0
max_inliers = 0
best_line = np.zeros((3, 1))

# Data points in homogeneous coordinates
points_h = np.vstack((x, y, np.ones((int(m)))))

while N > sample_count:
    # Pick two random samples
    samples = np.random.choice(np.arange(len(x)), 2, replace=False)
    id1 = samples[0]  # sample id 1
    id2 = samples[1]  # sample id 2

    # Determine the line crossing the points with the cross product of the points (in homogeneous coordinates).
    # Also normalize the line by dividing each element by sqrt(a^2+b^2), where a and b are the line coefficients

    ##-your-code-starts-here-##
    pt1 = points_h[:, id1]
    pt2 = points_h[:, id2]
    l = np.cross(pt1, pt2)
    l = l / np.linalg.norm(l[:2])
    ##-your-code-ends-here-##

    # Determine inliers by finding the indices for the line and data point dot
    # products (absolute value) that are less than inlier distance threshold.
    # Hint: point-to-line distance.

    ##-your-code-starts-here-##
    distances = np.abs(np.dot(points_h.T, l))
    inliers = np.where(distances < t)[0]
    ##-your-code-ends-here-##

    # Store the line in best_line and update max_inliers if the number of 
    # inliers is the best so far
    inlier_count = np.size(inliers)
    if inlier_count > max_inliers:
        best_line = l
        max_inliers = inlier_count

    # Update the estimate of the outlier ratio
    e = 1 - inlier_count / m
    # Update also the estimate for the required number of samples
    N = np.log(1 - p) / np.log(1 - (1 - e) ** s)

    sample_count += 1

# Least squares fitting to the inliers of the best hypothesis, i.e
# find the inliers similarly as above but this time for the best line.

##-your-code-starts-here-##
inliers = np.where(np.abs(np.dot(points_h.T, best_line)) < t)[0]
x_inliers = x[inliers]
y_inliers = y[inliers]
##-your-code-ends-here-##  

# Fit a line to the given points (non-homogeneous)
l = linefitlsq(x_inliers, y_inliers)

# Plot the resulting line and the inliers
k = -l[0] / l[1]
b = -l[2] / l[1]
plt.plot(np.arange(1, 101), k * np.arange(1, 101) + b, 'm-')
plt.plot(x[inliers], y[inliers], 'ro', markersize=7)
plt.show()
