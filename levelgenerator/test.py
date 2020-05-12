from copy import deepcopy
import random
from matplotlib import pyplot
import combine_structures


from math import sqrt, ceil, atan, atan2, cos, sin, pi, degrees, radians, tan
x = [1, 2, 3, 4, 5]
random.shuffle(x)
print(x)



release_point = combine_structures.find_release_point(pi/3)
trajectory = combine_structures.find_trajectory(release_point[0], release_point[1])
pyplot.plot(trajectory)

