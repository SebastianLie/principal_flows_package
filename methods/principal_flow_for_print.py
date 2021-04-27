import numpy as np
import matplotlib.pyplot as plt
from principal_flow import *
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D


def principal_flow_print(data, dimension, epsilon, h, flow_num=1, start_point=None, \
    kernel_type="identity", tol=1e-2, max_iter=40):
    # note: non-default arguments must be placed before default
    """ Computes the principal flow of the dataset. 
    Idea: This is a "greedy" implmentation of the principal flow 
    algorithm, developed originally by Professor Yao Zhi Gang.

    Defn: The Principal Flow of the dataset is basically an integral curve 
    that is always tangent to the direction of 'maximal variation' at any given point it contains.
    
    We assume the underlying structure of the data is a hypersphere.
    Starting from the centroid of the data set (user defined or calculated below),
    we apply the following procedure: 

    1. Project the data residing on the hypersphere onto the hyperplane 
    tangent to the centroid of the data (p). We use the log map of p, 
    obtaining a matrix of vectors on the tangent plane that point from p
    to the projected points. These are the plane vectors.

    2. Compute the largest principal component of the points using the plane vectors, 
    applying weights as necessary via the kernel function provided.

    3. The largest principal component is the new directions that the principal flow moves in. 
    We determine it's sign by making sure it is moving in the 
    same direction as the previous principal direction.

    4. We take a small step in each direction on the plane,
    then project it back to the hypersphere.

    5. We do 1-4 for the point on the opposite end of the growing principal flow.

    6. Store both points. 
    
    7. Repeat 1-6 until max_iter is reached.

    Args:
        data (np.array, (n,p)): [The data set, of shape (n,p), n = number of data points, p = dimension.]

        dimension (integer): dimension of data

        epsilon (float): [step size for the principal flow.]

        h (float): [Scale. Determines how "local" the principal flow is.
        Smaller scale => smaller neighbourhood, more emphasis on smaller pool of nearer points
        Bigger scale => bigger neighbourhood, emphasis on larger pool of points.]
        
        start_point (np.array, (p,1)): [the centroid, or the place to start the principal flow.
        Defaults to None.]

        kernel_type (string): [specifies the kernel function. Default is the identity kernel,
        which applies a weight of 1 to every point.]

        tol (float, optional): [useless for now.]

        max_iter (integer): number of points on each side of the flow.

    Returns:
        np.array: An array that contains the points of the principal flow.
    """
    # Sphere things
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    # handle data
    data = np.array(data)
    if data.shape[1] != dimension:
        data = data.T
    points_on_sphere = data
    
    # handle starting point
    if type(start_point) == None:
        p = sphere_centroid_finder_vecs(data, 3, 0.05, 0.01)
    else:
        # error report: for checking 
        assert type(start_point) is not np.array or \
            type(start_point) is not np.ndarray, "Start point must be an np.array or an np.ndarray"
        p = start_point

    flow = np.array(p)
    
    # handle kernel
    kernel_functions = {"binary": binary_kernel, "gaussian": gaussian_kernel, "identity": identity_kernel}
    assert kernel_type in kernel_functions.keys(), "Kernel must be binary, gaussian or identity."
    kernel = kernel_functions[kernel_type]

    # handle scale of kernel
    # TODO calculate range of h's and give a value for reconstructing data relatively accurately.
    # for now h is input

    p_opp = p
    num_iter = 0
    while True:
        print(num_iter)
        num_iter += 1
        if num_iter == 1:
            weights = kernel(h, points_on_sphere, p)
            not_in_hood = []
            in_hood = []
            for a in points_on_sphere:
                if np.linalg.norm(p - a) > h:
                    not_in_hood.append(a)
                else:
                    in_hood.append(a)
            in_hood = np.array(in_hood)
            not_in_hood = np.array(not_in_hood)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            plane_vectors2 = np.array(list(map(lambda point: log_map_sphere(p, point), in_hood)))
            vx, vy, vz = plane_vectors2.T
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
            ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot
            ax.quiver(p[0], p[1], p[2], vx, vy, vz, arrow_length_ratio=0.5)
            xx, yy, zz = not_in_hood.T
            nx, ny, nz = in_hood.T
            ax.scatter(xx, yy, zz, color="k", s=30)
            ax.scatter(nx, ny, nz, color="g", s=30)
            ax.scatter(p[0], p[1], p[2], color="r",s=50)
            plt.show()
            try:
                _, principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, weights, component=flow_num)
            except ValueError as err:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            principal_direction_opp = - principal_direction

            fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
            ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot
            xx, yy, zz = data.T
            px, py, pz = principal_direction
            ax.scatter(xx, yy, zz, color="k", s=10)
            ax.scatter(p[0], p[1], p[2], color="r", s=50)
            ax.quiver(p[0], p[1], p[2], px, py, pz, color="m", length=0.5, arrow_length_ratio=0.7)
            ax.quiver(p[0], p[1], p[2], -px, -py, -pz, color="m", length=0.5, arrow_length_ratio=0.7)
            plt.show()
        
        else:
            # calculate for one direction, then the other 
            weights = kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            past_direction = principal_direction
            try:
                _, principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, weights, component=flow_num)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            # make sure same direction
            if angle(past_direction, principal_direction) > math.pi/2:
                principal_direction = -principal_direction
            
            weights_opp = kernel(h, points_on_sphere, p_opp)
            plane_vectors_opp = np.array(list(map(lambda point: log_map_sphere(p_opp, point), points_on_sphere)))
            past_direction_opp = principal_direction_opp
            try:
                _, principal_direction_opp = compute_principal_component_vecs_weighted(plane_vectors_opp, p, weights_opp, component=flow_num)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            # make sure same direcwtion
            if angle(past_direction_opp, principal_direction_opp) > math.pi/2:
                principal_direction_opp = -principal_direction_opp
            print(p_opp)
            print(p)
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
            ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot
            xx, yy, zz = data.T
            px, py, pz = principal_direction
            opx, opy, opz = principal_direction_opp
            ax.scatter(xx, yy, zz, color="k", s=10)
            flow_temp = np.reshape(flow, (-1, dimension))
            ax.scatter(flow_temp.T[0], flow_temp.T[1], flow_temp.T[2], color="r", s=50)
            ax.quiver(p[0], p[1], p[2], px, py, pz, color="m", length=0.5, arrow_length_ratio=0.7)
            ax.quiver(p_opp[0], p_opp[1], p_opp[2], opx, opy, opz, color="m", length=0.5, arrow_length_ratio=0.7)
            plt.show()

        # update 1 direction
        p_prime_plane = p + epsilon * principal_direction
        p_prime = exp_map_sphere(p, p_prime_plane - p)
        p = p_prime

        # then the other
        p_prime_plane_opp = p_opp + epsilon * principal_direction_opp
        p_prime_opp = exp_map_sphere(p_opp, p_prime_plane_opp - p_opp)
        p_opp = p_prime_opp

        # now add to the curve
        flow = np.concatenate((flow, p))
        flow = np.concatenate((flow, p_opp))


        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
        ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot
        flow_toprint = np.reshape(flow, (-1, dimension))
        x_curve, y_curve, z_curve = flow_toprint.T
        xx, yy, zz = data.T
        ax.scatter(xx, yy, zz, color="k", s=10)
        ax.scatter(x_curve, y_curve, z_curve, color="r", s=50)

        #plt.savefig("manual_flow_pics/{}.".format(num_iter))
        plt.show()
        if num_iter > max_iter:
            break
        
    flow = np.reshape(flow, (-1, dimension))

    return flow

def principal_flow_print_each_iter(data, dimension, epsilon, h, flow_num=1, start_point=None, \
    kernel_type="identity", tol=1e-2, max_iter=40):
    # note: non-default arguments must be placed before default
    """ Computes the principal flow of the dataset. 
    Idea: This is a "greedy" implmentation of the principal flow 
    algorithm, developed originally by Professor Yao Zhi Gang.

    Defn: The Principal Flow of the dataset is basically an integral curve 
    that is always tangent to the direction of 'maximal variation' at any given point it contains.
    
    We assume the underlying structure of the data is a hypersphere.
    Starting from the centroid of the data set (user defined or calculated below),
    we apply the following procedure: 

    1. Project the data residing on the hypersphere onto the hyperplane 
    tangent to the centroid of the data (p). We use the log map of p, 
    obtaining a matrix of vectors on the tangent plane that point from p
    to the projected points. These are the plane vectors.

    2. Compute the largest principal component of the points using the plane vectors, 
    applying weights as necessary via the kernel function provided.

    3. The largest principal component is the new directions that the principal flow moves in. 
    We determine it's sign by making sure it is moving in the 
    same direction as the previous principal direction.

    4. We take a small step in each direction on the plane,
    then project it back to the hypersphere.

    5. We do 1-4 for the point on the opposite end of the growing principal flow.

    6. Store both points. 
    
    7. Repeat 1-6 until max_iter is reached.

    Args:
        data (np.array, (n,p)): [The data set, of shape (n,p), n = number of data points, p = dimension.]

        dimension (integer): dimension of data

        epsilon (float): [step size for the principal flow.]

        h (float): [Scale. Determines how "local" the principal flow is.
        Smaller scale => smaller neighbourhood, more emphasis on smaller pool of nearer points
        Bigger scale => bigger neighbourhood, emphasis on larger pool of points.]
        
        start_point (np.array, (p,1)): [the centroid, or the place to start the principal flow.
        Defaults to None.]

        kernel_type (string): [specifies the kernel function. Default is the identity kernel,
        which applies a weight of 1 to every point.]

        tol (float, optional): [useless for now.]

        max_iter (integer): number of points on each side of the flow.

    Returns:
        np.array: An array that contains the points of the principal flow.
    """
    # Sphere things
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    # handle data
    data = np.array(data)
    if data.shape[1] != dimension:
        data = data.T
    points_on_sphere = data
    
    # handle starting point
    if type(start_point) == None:
        p = sphere_centroid_finder_vecs(data, 3, 0.05, 0.01)
    else:
        # error report: for checking 
        assert type(start_point) is not np.array or \
            type(start_point) is not np.ndarray, "Start point must be an np.array or an np.ndarray"
        p = start_point

    flow = np.array(p)
    
    # handle kernel
    kernel_functions = {"binary": binary_kernel, "gaussian": gaussian_kernel, "identity": identity_kernel}
    assert kernel_type in kernel_functions.keys(), "Kernel must be binary, gaussian or identity."
    kernel = kernel_functions[kernel_type]

    # handle scale of kernel
    # TODO calculate range of h's and give a value for reconstructing data relatively accurately.
    # for now h is input

    p_opp = p
    num_iter = 0
    while True:
        print(num_iter)
        num_iter += 1
        if num_iter == 1:
            weights = kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            try:
                _, principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, weights, component=flow_num)
            except ValueError as err:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            principal_direction_opp = - principal_direction        
        else:
            # calculate for one direction, then the other 
            weights = kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            past_direction = principal_direction
            try:
                _, principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, weights, component=flow_num)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            # make sure same direction
            if angle(past_direction, principal_direction) > math.pi/2:
                principal_direction = -principal_direction
            
            weights_opp = kernel(h, points_on_sphere, p_opp)
            plane_vectors_opp = np.array(list(map(lambda point: log_map_sphere(p_opp, point), points_on_sphere)))
            past_direction_opp = principal_direction_opp
            try:
                _, principal_direction_opp = compute_principal_component_vecs_weighted(plane_vectors_opp, p, weights_opp, component=flow_num)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            # make sure same direcwtion
            if angle(past_direction_opp, principal_direction_opp) > math.pi/2:
                principal_direction_opp = -principal_direction_opp
            
        # update 1 direction
        p_prime_plane = p + epsilon * principal_direction
        p_prime = exp_map_sphere(p, p_prime_plane - p)
        p = p_prime

        # then the other
        p_prime_plane_opp = p_opp + epsilon * principal_direction_opp
        p_prime_opp = exp_map_sphere(p_opp, p_prime_plane_opp - p_opp)
        p_opp = p_prime_opp

        # now add to the curve
        flow = np.concatenate((flow, p))
        flow = np.concatenate((flow, p_opp))

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
        ax.plot_surface(x, y, z, color='k', rstride=1, cstride=1, alpha=0.1) # alpha affects transparency of the plot
        flow_toprint = np.reshape(flow, (-1, dimension))
        x_curve, y_curve, z_curve = flow_toprint.T
        xx, yy, zz = data.T
        ax.scatter(xx, yy, zz, color="k", s=30)
        ax.view_init(elev=80., azim=150)
        ax.scatter(x_curve, y_curve, z_curve, color="r", s=50)
        plt.savefig("flow_pics/{}.".format(num_iter))
        #plt.show()
        if num_iter > max_iter:
            break
        
    flow = np.reshape(flow, (-1, dimension))

    return flow


'''
data = pd.read_csv('Sample_Data/data5.csv')
data_np = data.to_numpy()

data_np = (data_np.T[1:]).T

final_p = sphere_centroid_finder_vecs(data_np, 3, 0.05, 0.01, max_iter=200)
h = choose_h_binary(data_np.T, final_p, 30) # needs to be very high!
curve = principal_flow_print(data_np, 3, 0.05, h, flow_num=1, start_point=final_p, kernel_type="binary", max_iter=2)
'''


def principal_flow_directions(data, dimension, epsilon, h, flow_num=1, start_point=None, \
    kernel_type="identity", tol=1e-2, max_iter=40):
    # note: non-default arguments must be placed before default
    """ Computes the principal flow of the dataset. 
    Idea: This is a "greedy" implmentation of the principal flow 
    algorithm, developed originally by Professor Yao Zhi Gang.

    Defn: The Principal Flow of the dataset is basically an integral curve 
    that is always tangent to the direction of 'maximal variation' at any given point it contains.
    
    We assume the underlying structure of the data is a hypersphere.
    Starting from the centroid of the data set (user defined or calculated below),
    we apply the following procedure: 

    1. Project the data residing on the hypersphere onto the hyperplane 
    tangent to the centroid of the data (p). We use the log map of p, 
    obtaining a matrix of vectors on the tangent plane that point from p
    to the projected points. These are the plane vectors.

    2. Compute the largest principal component of the points using the plane vectors, 
    applying weights as necessary via the kernel function provided.

    3. The largest principal component is the new directions that the principal flow moves in. 
    We determine it's sign by making sure it is moving in the 
    same direction as the previous principal direction.

    4. We take a small step in each direction on the plane,
    then project it back to the hypersphere.

    5. We do 1-4 for the point on the opposite end of the growing principal flow.

    6. Store both points. 
    
    7. Repeat 1-6 until max_iter is reached.

    Args:
        data (np.array, (n,p)): [The data set, of shape (n,p), n = number of data points, p = dimension.]

        dimension (integer): dimension of data

        epsilon (float): [step size for the principal flow.]

        h (float): [Scale. Determines how "local" the principal flow is.
        Smaller scale => smaller neighbourhood, more emphasis on smaller pool of nearer points
        Bigger scale => bigger neighbourhood, emphasis on larger pool of points.]
        
        start_point (np.array, (p,1)): [the centroid, or the place to start the principal flow.
        Defaults to None.]

        kernel_type (string): [specifies the kernel function. Default is the identity kernel,
        which applies a weight of 1 to every point.]

        tol (float, optional): [useless for now.]

        max_iter (integer): number of points on each side of the flow.

    Returns:
        np.array: An array that contains the points of the principal flow.
    """
    # Sphere things
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    # handle data
    data = np.array(data)
    if data.shape[1] != dimension:
        data = data.T
    points_on_sphere = data
    
    # handle starting point
    if type(start_point) == None:
        p = sphere_centroid_finder_vecs(data, 3, 0.05, 0.01)
    else:
        # error report: for checking 
        assert type(start_point) is not np.array or \
            type(start_point) is not np.ndarray, "Start point must be an np.array or an np.ndarray"
        p = start_point

    flow = np.array([])
    directions = []
    # handle kernel
    kernel_functions = {"binary": binary_kernel, "gaussian": gaussian_kernel, "identity": identity_kernel}
    assert kernel_type in kernel_functions.keys(), "Kernel must be binary, gaussian or identity."
    kernel = kernel_functions[kernel_type]

    p_opp = p
    num_iter = 0
    while True:
        flow = np.concatenate((flow, p))
        flow = np.concatenate((flow, p_opp))
        print(num_iter)
        num_iter += 1
        if num_iter == 1:
            weights = kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            try:
                _, principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, weights, component=flow_num)
            except ValueError as err:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            principal_direction_opp = - principal_direction
        else:
            # calculate for one direction, then the other 
            weights = kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            past_direction = principal_direction
            try:
                _, principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, weights, component=flow_num)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            # make sure same direction
            if angle(past_direction, principal_direction) > math.pi/2:
                principal_direction = -principal_direction
            
            weights_opp = kernel(h, points_on_sphere, p_opp)
            plane_vectors_opp = np.array(list(map(lambda point: log_map_sphere(p_opp, point), points_on_sphere)))
            past_direction_opp = principal_direction_opp
            try:
                _, principal_direction_opp = compute_principal_component_vecs_weighted(plane_vectors_opp, p, weights_opp, component=flow_num)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            # make sure same direcwtion
            if angle(past_direction_opp, principal_direction_opp) > math.pi/2:
                principal_direction_opp = -principal_direction_opp
        # update 1 direction
        p_prime_plane = p + epsilon * principal_direction
        p_prime = exp_map_sphere(p, p_prime_plane - p)
        p = p_prime

        # then the other
        p_prime_plane_opp = p_opp + epsilon * principal_direction_opp
        p_prime_opp = exp_map_sphere(p_opp, p_prime_plane_opp - p_opp)
        p_opp = p_prime_opp

        # now add to the curve
        
        directions.append(principal_direction)
        directions.append(principal_direction_opp)

        if num_iter > max_iter:
            break
        
    flow = np.reshape(flow, (-1, dimension))

    return flow, np.array(directions)
