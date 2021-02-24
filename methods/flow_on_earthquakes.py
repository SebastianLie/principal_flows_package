import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from common_methods_sphere import put_on_sphere

# https://stackoverflow.com/questions/10473852/convert-latitude-and-longitude-to-point-in-3d-space
def LLHtoECEF(arr):
    # see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html
    lat, lon, alt = arr
    rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
    f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
    cosLat = np.cos(lat)
    sinLat = np.sin(lat)
    FF     = (1.0-f)**2
    C      = 1/np.sqrt(cosLat**2 + FF * sinLat**2)
    S      = C * FF

    x = (rad * C + alt)*cosLat * np.cos(lon)
    y = (rad * C + alt)*cosLat * np.sin(lon)
    z = (rad * S + alt)*sinLat

    return np.array([x, y, z])

print(os.path.abspath(os.curdir))
data = pd.read_csv("earthquakes.csv")
# print(data.columns)
data = data[data["Magnitude"] > 8]
data = data.filter(items=['Latitude', 'Longitude', 'Depth'])

data_np = data.values
print(data_np.shape)
data_on_sphere = put_on_sphere(np.array(list(map(LLHtoECEF, data_np))))

phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot
xx, yy, zz = data_on_sphere.T
ax.scatter(xx, yy, zz, color="k", s=50)
plt.show()






