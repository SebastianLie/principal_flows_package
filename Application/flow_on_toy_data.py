import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common_methods_sphere import *
from principal_flow import *
from principal_flow_for_print import *
from principal_boundary_flows import principal_boundary
from centroid_finder import *
from schilds_ladder import *
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

def noising_data(points, variance=0.1, seed=999):
    # approach: add a random centered normal RV to every dimension
    # of each point
    # still needs to be normalised to have norm 1!
    # data not usable after this step yet
    # assumes points are in (n,p) format
    np.random.seed(seed)
    p = points.shape[1]
    sigma = math.sqrt(variance)
    return np.array([np.array(point) - np.random.normal(0, sigma, p) for point in points])

def testing_flow_noisy():
    data = pd.read_csv('Sample_Data/data13.csv')
    data_np = data.to_numpy()
    data_np = data_np.T[1:]
    noisy_data = noising_data(data_np, 0.01, seed=88)
    noisy_data = put_on_sphere(noisy_data)
    final_p = sphere_centroid_finder_vecs(noisy_data, 3, 0.05, 0.01)
    #print(final_p)
    #h = choose_h_gaussian(noisy_data, final_p, 50) # needs to be very high! # needs to be very high!
    h = choose_h_binary(noisy_data, final_p, 25) # needs to be very high! # needs to be very high!
    curve = principal_flow_print(noisy_data, 3, 0.02, h, flow_num=1, start_point=final_p, kernel_type="binary", max_iter=30)
    #print(curve)
    x_curve, y_curve, z_curve = curve.T

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot

    xx, yy, zz = noisy_data.T
    ax.scatter(xx, yy, zz, color="k", s=50)
    ax.scatter(x_curve, y_curve, z_curve, color="r", s=50)

    plt.show()
    
def testing_flow_single():
    data = pd.read_csv('Sample_Data/data13.csv')
    data_np = data.to_numpy()
    
    data_np = (data_np.T[1:]).T
    random.seed(999)
    final_p = sphere_centroid_finder_vecs(data_np, 3, 0.05, 0.01)
    #print(final_p)
    print(choose_h_gaussian(data_np.T, final_p, 100))
    print(choose_h_gaussian(data_np.T, final_p, 1))
    h = choose_h_binary(data_np.T, final_p, 10) # needs to be very high!
    curve = principal_flow_print_each_iter(data_np, 3, 0.02, h, start_point=final_p, kernel_type="binary", max_iter=30)
    #print(curve)
    x_curve, y_curve, z_curve = curve.T
    xd, yd, zd = directions.T
   
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot

    xx, yy, zz = data_np
    ax.scatter(xx, yy, zz, color="k", s=5)
    ax.quiver(x_curve, y_curve, z_curve, xd, yd, zd, color="r", length=0.1, arrow_length_ratio=0.)

    plt.show()

testing_flow_single()
#testing_flow_noisy()

def testing_boundary_flow_noisy():
    data = pd.read_csv('Sample_Data/data13.csv')
    data_np = data.to_numpy()
    data_np = data_np.T[1:]
    noisy_data = noising_data(data_np, 0.03, seed=998)  # good seed, don't change!!
    noisy_data = put_on_sphere(noisy_data)
    final_p = sphere_centroid_finder_vecs(noisy_data, 3, 0.05, 0.01)
    #print(final_p)
    h = choose_h_gaussian(noisy_data, final_p, 50) # needs to be very high! # needs to be very high!
    radius = choose_h_binary(noisy_data, final_p, 40)
    upper, curve, lower = principal_boundary(noisy_data, 3, 0.02, h, radius, start_point=final_p, kernel_type="gaussian", max_iter=10)
    #print(curve)
    x_upper, y_upper, z_upper = upper.T
    x_curve, y_curve, z_curve = curve.T
    x_lower, y_lower, z_lower = lower.T

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot
    xx, yy, zz = noisy_data.T
    ax.scatter(xx, yy, zz, color="k", s=50)
    ax.scatter(x_curve, y_curve, z_curve, color="r", s=50)
    ax.scatter(x_upper, y_upper, z_upper, color="b", s=50)
    ax.scatter(x_lower, y_lower, z_lower, color="g", s=50)

    plt.show()

def testing_boundary_flow():
    data = pd.read_csv('Sample_Data/data5.csv')
    data_np = data.to_numpy()
    data_np = data_np.T[1:]
    final_p = sphere_centroid_finder_vecs(data_np, 3, 0.05, 0.01)
    #print(final_p)
    h = choose_h_gaussian(data_np, final_p, 20) # needs to be very high! # needs to be very high!
    radius = choose_h_binary(data_np, final_p, 50)
    upper, curve, lower = principal_boundary(data_np, 3, 0.02, h, radius, start_point=final_p, kernel_type="gaussian", max_iter=10)
    #print(curve)
    x_upper, y_upper, z_upper = upper.T
    x_curve, y_curve, z_curve = curve.T
    x_lower, y_lower, z_lower = lower.T

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot
    xx, yy, zz = data_np.T
    ax.scatter(xx, yy, zz, color="k", s=50)
    ax.scatter(x_curve, y_curve, z_curve, color="r", s=50)
    ax.scatter(x_upper, y_upper, z_upper, color="b", s=50)
    ax.scatter(x_lower, y_lower, z_lower, color="g", s=50)

    plt.show()

#testing_boundary_flow()