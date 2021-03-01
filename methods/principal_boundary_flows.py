import math
import numpy as np
from principal_flow import compute_principal_component_vecs_weighted, \
    binary_kernel, gaussian_kernel, identity_kernel
from common_methods_sphere import log_map_sphere, exp_map_sphere, angle
from centroid_finder import sphere_centroid_finder_vecs
from schilds_ladder import schilds_ladder_hypersphere

# radius use binary kernel h
# plotting boundary: need to "transport" 1st eigenvector at p on the flow to p'
# on the boundary, then plot these vectors
# idea: use plot to plot the flow as a line
def principal_boundary(data, dimension, epsilon, h, radius, start_point=None, \
    kernel_type="identity", max_iter=40, parallel_transport=False):
    # points on sphere now!!
    # note: non-default arguments must be placed before default
    """ Computes the principal boundary of the dataset.
    Idea: This is a "greedy" implmentation of the principal boundary
    algorithm, developed originally by Professor Yao Zhi Gang.
    
    It uses the greedy version of the pricipal flow algorithm.

    Note: seems to work now!

    Args:
        data (np.array, (n,p)): [The data set, of shape (n,p), n = number of data points, p = dimension.]

        dimension (integer): [dimension of data]

        epsilon (float): [step size for the principal flow.]

        radius (float): [radius for boundary to move. use the function choose_h_binary to set
        the distance it should move that takes n% of the points into consideration]

        h (float): [Scale. Determines how "local" the principal flow is. 
        Smaller scale => smaller neighbourhood, more emphasis on smaller pool of nearer points
        Bigger scale => bigger neighbourhood, emphasis on larger pool of points.]
        
        start_point (np.array, (p,1)): [the centroid, or the place to start the principal flow. 
        Defaults to None.]

        kernel_type (string): [specifies the kernel function. Default is the identity kernel, 
        which applies a weight of 1 to every point.]

        tol (float, optional): [useless for now.] (use as max of the min distance from flow
        to data points? Potential stopping criterion?)

        max_iter (float, optional): [controls the amount of points]

    Returns:
        np.array: An array that contains the points of the principal flow.
    """
    data = np.array(data)
    if data.shape[1] != dimension:
        data = data.T
    
    # handle starting point
    if type(start_point) == None:
        p = sphere_centroid_finder_vecs(data, 3, 0.05, 0.01)
    else:
        # error report: for checking 
        assert type(start_point) is not np.array or \
            type(start_point) is not np.ndarray, "Start point must be an np.array or an np.ndarray"
        p = start_point
    
    upper_boundary = list()
    flow = np.array(p)
    lower_boundary = list()
    if parallel_transport:
        upper_vectors = list()
        lower_vectors = list()
    # handle kernel
    kernel_functions = {"binary": binary_kernel, "gaussian": gaussian_kernel, "identity": identity_kernel}
    assert kernel_type in kernel_functions.keys(), "Kernel must be binary, gaussian or identity."
    kernel = kernel_functions[kernel_type]

    p_opp = p
    num_iter = 0
    while True:
        print(num_iter)
        num_iter += 1
        if num_iter == 1:
            weights = kernel(h, data, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), data)))
            try:
                principal_pair, boundary_pair = compute_principal_component_vecs_weighted(\
                    plane_vectors, p, weights, boundary=True)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
            
            first_eigenval = principal_pair[0]
            second_eigenval = boundary_pair[0]

            # for boundary
            past_orthogonal = boundary_pair[1]

            # for flow
            principal_direction = principal_pair[1]
            principal_direction_opp = - principal_direction

            # update boundary
            sigma_f_p = second_eigenval/first_eigenval * radius  # how much to move for boundary

            upper_boundary_point_plane = p + sigma_f_p * past_orthogonal
            upper_boundary_point = exp_map_sphere(p, upper_boundary_point_plane - p)
            upper_boundary.append(upper_boundary_point)

            if parallel_transport:
                transported_vector = schilds_ladder_hypersphere(p, upper_boundary_point, principal_direction)
                upper_vectors.append(transported_vector)

            lower_boundary_point_plane = p - sigma_f_p * past_orthogonal
            lower_boundary_point = exp_map_sphere(p, lower_boundary_point_plane - p)
            lower_boundary.append(lower_boundary_point)

            if parallel_transport:
                transported_vector = schilds_ladder_hypersphere(p, lower_boundary_point, principal_direction)
                lower_vectors.append(transported_vector)

            # first direction
            p_prime_plane = p + epsilon * principal_direction
            p_prime = exp_map_sphere(p, p_prime_plane - p)
            p = p_prime

            # now we do the other direction
            p_prime_plane_opp = p_opp + epsilon * principal_direction_opp
            p_prime_opp = exp_map_sphere(p_opp, p_prime_plane_opp - p_opp)
            p_opp = p_prime_opp

        else:
            # calculate for one direction, then the other 
            weights = kernel(h,data, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), data)))
            past_direction = principal_direction
            try:
                principal_pair, boundary_pair = compute_principal_component_vecs_weighted(\
                    plane_vectors, p, weights, boundary=True)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
           
           # obtain boundary for this point - first we obtain intial info
            first_eigenval = principal_pair[0]
            second_eigenval = boundary_pair[0]
            orthogonal_to_flow = boundary_pair[1]
            if angle(orthogonal_to_flow, past_orthogonal) > math.pi/2:
                orthogonal_to_flow = - orthogonal_to_flow
            
            # Get principal direction
            principal_direction = principal_pair[1]
            if angle(past_direction, principal_direction) > math.pi/2:
                principal_direction = - principal_direction
        
            # move in direction orthogonal to flow, a distance of sigma_f_p
            sigma_f_p = second_eigenval/first_eigenval * radius

            # get both sides of the boundary + and - orthogonal_to_flow
            upper_boundary_point_plane = p + sigma_f_p * orthogonal_to_flow
            upper_boundary_point = exp_map_sphere(p, upper_boundary_point_plane - p)
            upper_boundary.append(upper_boundary_point)
            if parallel_transport:
                transported_vector = schilds_ladder_hypersphere(p, upper_boundary_point, principal_direction)
                upper_vectors.append(transported_vector)                

            lower_boundary_point_plane = p - sigma_f_p * orthogonal_to_flow
            lower_boundary_point = exp_map_sphere(p, lower_boundary_point_plane - p)
            lower_boundary.append(lower_boundary_point)
            if parallel_transport:
                transported_vector = schilds_ladder_hypersphere(p, lower_boundary_point, principal_direction)
                lower_vectors.append(transported_vector)                

            past_orthogonal = orthogonal_to_flow # always updated only for upper, so past is the benchmark for upper.
            
            # Next we update the main point for the flow:
            # update point p
            p_prime_plane = p + epsilon * principal_direction
            p_prime = exp_map_sphere(p, p_prime_plane - p)
            p = p_prime
           
            weights_opp = kernel(h, data, p_opp)
            plane_vectors_opp = np.array(list(map(lambda point: log_map_sphere(p_opp, point), data)))
            past_direction_opp = principal_direction_opp
            try:
                principal_pair_opp, boundary_pair_opp = compute_principal_component_vecs_weighted(\
                   plane_vectors_opp, p, weights_opp, boundary=True)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break

            # get info again
            first_eigenval_opp = principal_pair_opp[0]
            second_eigenval_opp = boundary_pair_opp[0]
            orthogonal_to_flow_opp = boundary_pair_opp[1]

            if angle(orthogonal_to_flow_opp, past_orthogonal) > math.pi/2:
                orthogonal_to_flow_opp = - orthogonal_to_flow_opp

            # make sure same direction
            principal_direction_opp = principal_pair_opp[1]
            if angle(past_direction_opp, principal_direction_opp) > math.pi/2:
                principal_direction_opp = - principal_direction_opp
          
            sigma_f_p_opp = second_eigenval_opp/first_eigenval_opp * radius

            upper_boundary_point_opp_plane = p_opp + sigma_f_p_opp * orthogonal_to_flow_opp
            upper_boundary_point_opp = exp_map_sphere(p_opp, upper_boundary_point_opp_plane - p_opp)
            upper_boundary.append(upper_boundary_point_opp)

            if parallel_transport:
                transported_vector = schilds_ladder_hypersphere(p_opp, upper_boundary_point_opp, principal_direction_opp)
                upper_vectors.append(transported_vector)                

            lower_boundary_point_opp_plane = p_opp - sigma_f_p_opp * orthogonal_to_flow_opp
            lower_boundary_point_opp = exp_map_sphere(p_opp, lower_boundary_point_opp_plane - p_opp)
            lower_boundary.append(lower_boundary_point_opp)
            if parallel_transport:
                transported_vector = schilds_ladder_hypersphere(p_opp, lower_boundary_point_opp, principal_direction_opp)
                lower_vectors.append(transported_vector)               

            # now we do the other direction
            p_prime_plane_opp = p_opp + epsilon * principal_direction_opp
            p_prime_opp = exp_map_sphere(p_opp, p_prime_plane_opp - p_opp)
            p_opp = p_prime_opp

            # now add to the curve
            flow = np.concatenate((flow, p))
            flow = np.concatenate((p_opp, flow))

        if num_iter >= max_iter:
            break
    flow = np.reshape(flow, (-1, dimension))
    if parallel_transport:
        return np.array(upper_boundary), flow, np.array(lower_boundary), np.array(upper_vectors), np.array(lower_vectors)
    else:
        return np.array(upper_boundary), flow, np.array(lower_boundary)
