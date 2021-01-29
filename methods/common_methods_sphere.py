import numpy as np


#####################
# Generating Points #
#####################


def sample_spherical_evenly_distributed(npoints, ndim=3) -> np.array:
    # plot random points on the sphere
    # from https://mathworld.wolfram.com/SpherePointPicking.html

    '''
    Idea here is that we are generating
    random points on a unit sphere.

    So the points have to be at a dist of 1 from origin
    Therefore, we simply randomly generate vectors and divide them by the 
    norm of the vector so that the norm of the vector created = 1.
    '''
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sample_spherical_denser(npoints):
    '''
    Idea is to generate spherical polar coordinates, 
    then convert them to cartesian/rectangular
    fix rho = 1 to ensure that points are always on the unit sphere.
    rho is the distance between the point and the origin.
    0 <= phi <= pi
    0 <= theta <= 2pi
    ideas kind of inspired by 
    http://corysimon.github.io/articles/uniformdistn-on-sphere/
    '''
    rho = np.ones(npoints)
    pi = np.pi
    phi = np.random.uniform(low=pi/4, high=pi/2, size=npoints)
    theta = np.random.uniform(low=pi, high=3*pi/2, size=npoints)
    spherical_coords = np.array([rho, phi, theta])

    return spherical_coords

def generate_points(npoints):
    '''
    TODO: MAKE THIS NON RANDOM - FIXED OFFSET
    '''
    rho = np.ones(npoints)
    pi = np.pi
    orig_phi = np.random.uniform(low=pi/4, high=pi/2)
    orig_theta = np.random.uniform(low=pi, high=3*pi/2)
    phi = []
    theta = []
    for i in range(npoints):
        indx = i+1
        phi.append(orig_phi-indx*0.2)
        theta.append(orig_theta+indx*0.2)
    spherical_coords = np.array([rho, phi, theta])

    return spherical_coords

def generate_square():
    rho = np.ones(4)
    pi = np.pi
    orig_phi = pi/4
    orig_theta = pi
    phi = []
    theta = []
    phi.append(orig_phi-0.25)
    theta.append(orig_theta+0.25)
    phi.append(orig_phi+0.25)
    theta.append(orig_theta-0.25)
    phi.append(orig_phi-0.25)
    theta.append(orig_theta-0.25)
    phi.append(orig_phi+0.25)
    theta.append(orig_theta+0.25)
    return np.array([rho, phi, theta])

def generate_dense_square():
    rho = np.ones(10)
    pi = np.pi
    orig_phi = pi/4
    orig_theta = pi
    phi = []
    theta = []
    phi.append(orig_phi-0.25)
    theta.append(orig_theta+0.25)
    phi.append(orig_phi-0.25)
    theta.append(orig_theta+0.2)
    phi.append(orig_phi+0.25)
    theta.append(orig_theta-0.25)
    phi.append(orig_phi+0.2)
    theta.append(orig_theta-0.25)
    phi.append(orig_phi-0.25)
    theta.append(orig_theta-0.25)
    phi.append(orig_phi-0.25)
    theta.append(orig_theta-0.2)
    phi.append(orig_phi-0.2)
    theta.append(orig_theta-0.25)
    phi.append(orig_phi+0.25)
    theta.append(orig_theta+0.2)
    phi.append(orig_phi+0.25)
    theta.append(orig_theta+0.25)
    phi.append(orig_phi+0.2)
    theta.append(orig_theta+0.25)
    return np.array([rho, phi, theta])

def generate_square_many(npoints):
    rho = np.ones(npoints*4)
    pi = np.pi
    orig_phi = pi/4
    orig_theta = pi
    phi = []
    theta = []
    for i in range(npoints):

        phi.append(orig_phi-0.25+i*0.01)
        theta.append(orig_theta+0.25+i*0.01)
        phi.append(orig_phi+0.25+i*0.01)
        theta.append(orig_theta-0.25+i*0.01)
        phi.append(orig_phi-0.25-i*0.01)
        theta.append(orig_theta-0.25-i*0.01)
        phi.append(orig_phi+0.25+i*0.01)
        theta.append(orig_theta+0.25+i*0.01)

    return np.array([rho, phi, theta])


####################
# Helper functions #
####################

def put_on_sphere(points):
    """ Puts an arbitrary number of points(vectors) 
    onto the hypersphere i.e normalising to 1 unit norm

    Args:
        points ([numpy.ndarray]): points to put on the hypersphere

    Returns:
        np.array: np.array containing the points residing on the hypersphere.
    """    
    return np.array([point/np.linalg.norm(point) for point in points])

def spherical_to_cartesian(point) -> list:
    '''
    assumes input is 3 dimensional, spherical coord, uses:
    https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,y2%2Bz2).
    '''
    rho, phi, theta = point
    return [rho*np.sin(phi)*np.cos(theta), rho*np.sin(phi)*np.sin(theta), rho*np.cos(phi)]

def get_pairwise_distances(points, p) -> list:
    '''
    Helper function to get sum of pairwise distances
    points = (n,p) array
    '''
    points = np.array(points)
    p = np.array(p)
    return list(map(lambda x: np.linalg.norm(x - p), points))

def get_sum_pairwise_distances(points, p) -> float:
    '''
    Helper function to get sum of pairwise distances
    '''
    points = np.array(points)
    p = np.array(p)
    return sum(list(map(lambda x: np.linalg.norm(x - p), points)))

def angle(x, y) -> float:
    '''
    Helper function to do calculate the angle between x and y,
    2 vectors in R^p
    '''
    x_y = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
    if abs(x_y) > 1:
        x_y = round(x_y)
    return np.arccos(x_y)

#####################
# Projecting points #
#####################

def exp_map_sphere(tangent_point, tangent_vector):
    '''
    Exp maps Usage - Projected point simply = exp_map(p, tangent_vector),
    where tangent vector = point on plane - p, p is base point for the tangent plane
    since point on plane - p is the vector that points from p, the base point of the tangent
    plane to the point we want to project.

    Args:
    tangent_point: 3D array of the point on the sphere 
    at which the tangent plane is tangent to.
    tangent_vector: a vector that indicates the direction from tangent_point
    to the point we want to project on the sphere.
     
    Formula from https://ronnybergmann.net/mvirt/manifolds/Sn/exp.html
    Checked against geomstats in python for correctness, checks out.
    '''
    x = tangent_point
    kaci = tangent_vector
    alpha = np.linalg.norm(kaci)/np.linalg.norm(x)
    if alpha == 0:
        return x
    else:
        return x*np.cos(alpha) + kaci/alpha * np.sin(alpha)


def log_map_sphere(tangent_point, sphere_point):
    '''
    Log maps Usage - for any point on the sphere s, 
    and a point at which the sphere is tangent to the plane 
    we wish to project on, t,
    log_map(t, s) gives the direction (vector) on THE TANGENT PLANE 
    in which t should move so that it becomes point closest to s 
    on the tangent plane defined by t.
    Thus, to obtain the projection of s onto the 
    tangent plane defined by t, 
    we add the tangent plane point, p and the logmap(p, point):
    p + log_map_sphere(p,point)

    Args:
    tangent_point: (p,1) array of the point on the sphere 
    at which the tangent plane is tangent to.
    sphere_point: the point on the sphere which we want to project onto the 
    tangent plane 

    formula from https://ronnybergmann.net/mvirt/manifolds/Sn/log.html
    Checked against geomstats in python for correctness, checks out.
    '''
    x = tangent_point
    y = sphere_point
    xy_angle = angle(x, y)
    if xy_angle == 0:
        return np.zeros(len(x))
    else:
        return xy_angle/np.sin(xy_angle) * (y - np.cos(xy_angle)*x)

################################
# Stopping Condition Functions #
################################

def test_centroid(points, centroid):
    '''
    Idea: To verify correctness of algorithm.
    Centroid found by algorithm must have the smallest sum of 
    pairwise distances between the points generated and itself
    compared to many randomly generated points: we use 1000 here.

    Returns True if can find a point that has a smaller sum of pairwise dist than 
    vs centroid.
    '''
    centroid_pairwise_dist_sum = get_sum_pairwise_distances(points, centroid)
    sample_points = sample_spherical_evenly_distributed(1000)
    sum_pairwise_dist_sample_points = np.array(list(map(lambda x: get_sum_pairwise_distances(points, x), sample_points)))

    return (sum_pairwise_dist_sample_points < centroid_pairwise_dist_sum).any()

def test_eig_diff(eig_values, tol):
    '''
    Idea:
    Stop when gaps between all the eigenvalues are v small.
    '''
    eig_values = np.array(eig_values)
    eig_values_from_1 = np.append(eig_values[1:], 0)
    gaps = (eig_values - eig_values_from_1)[:-1]
    return (gaps < tol).all()
