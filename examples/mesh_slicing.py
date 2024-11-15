import numpy as np
import trimesh as tm

mesh = tm.load_mesh("SimpleCoil.stl")
mesh.show()

debug = True
def get_intersection_lines(slice_height, structure):
        """Gets the intersections and z_levels.

        Returns:
            intersection_lines (list of arrays): A list of (n,2,2) arrays representing the intersection lines as start_node-end_node
            z_levels (array): Array of z levels corresponding to the intersections.
        """
        minz, maxz = structure.bounds[0, 2], structure.bounds[1, 2]
        # move a bit to avoid artifacts
        minz += 1e-3
        maxz -= 1e-3
        z_levels = np.arange(minz, maxz, slice_height)

        # define the slicing plane
        plane_normal = np.array((0.0, 0.0, 1.0))
        plane_orig = np.zeros(3).astype(float)

        intersection_lines, _, _ = tm.intersections.mesh_multiplane( structure,
            plane_orig, plane_normal, z_levels
        )
        # drop the empty intersections
        nonempty = np.array(len(inter) != 0 for inter in intersection_lines).astype(
            bool
        )
        z_levels = z_levels[nonempty].flatten()
        intersection_lines = [inter for inter in intersection_lines if len(inter) != 0]
        return intersection_lines, z_levels




def split_intersection(intersection):
    """Splits the intersection into connected components (branches).

    Args:
        intersection ((n,2,2)): array of intersection lines

    Returns:
        (m,) list of (k,2,2) arrays: intersections grouped into components
    """
    # group the same points in the intersections
    if debug==True: print(f"intersection shape: {intersection.shape}") ##Intersection is an array of n lines. The 2,2 are the start xy and end xy
    if debug==True: print(f" intersection piece: {intersection[0, :, :]}")
    if debug==True: print(f"intersection reshape: {intersection.reshape(-1,2).shape}") #The reshaping smashes together starts and ends (since the distinction doesn't matter for connectivity)
    if debug==True: print(f"intersection reshape piece: {intersection.reshape(-1,2)[0,:]}")
    grouped_rows = tm.grouping.group_rows(intersection.reshape(-1, 2))  # find line segments that share coordinates, i.e. they connect
    if debug==True: print(f"grouped rows: {grouped_rows}")
    if debug==True: print('\n')
    # assign each point an index.
    ''' 
    The grouped_rows is a list, which each element having len(n) for the n indices that share the same values
    
    example: [ [0,0], [1,1], [2,2], [0,0], [2,1] ]
    grouped_rows would output: [[0, 3], [1], [2], [4]]

    To get the indices that are shared, this loops through the grouped_rows and creates a numpy array with entries in the form [index of duplicate in original matrix, integer identifier of that group]
    
    so the second value is a label to identify which points are connected together. The points are at the array position given by the first number (index).
    
    
    in short, grouped_indices[:,0] is a list of all the indices that are repeated. grouped_indices[:, 1] is a list of labels that show which repeats are related to one another (grouped_indices[n, 1] == grouped_indices[m, 1] when n and m are the same value)
    
    '''
    grouped_indices = np.array(
        [[l, i] for i, ls in enumerate(grouped_rows) for l in ls]
    )
    # get the indices sorted so that i-th element of node_indices corresponds to the i-th point
    if debug==True: print(f"grouped_indices: {grouped_indices[:,0]}")
    
    """
    argsort returns an array of *indices* of the values that would make the array sequential. 
    
    for example:
    array = [0, 6, 2, 9]
    argsort_output = [0, 2, 1, 3]
    
    Here it takes the slice [:, 0] of grouped_indices, which corresponds to sorting by the value of the index. The result is a 1D array of indices of grouped_indices so that grouped_indices[arg] would be sequential.
    

    """
    
    try:
        arg = np.argsort(grouped_indices[:, 0])
    except IndexError:
        print(len(intersection))
        print(intersection)
        raise Exception()
    
    ### create node_indices, which are the values of the grouped_indices in sequential order
    
    """
    node_indices takes the indices to make grouped_indices sequential, determined by arg, and creates an array of the same order but containing the connectivity labels
    """
    node_indices = grouped_indices[arg, 1]
    if debug==True: print(f"node_indices shape: {node_indices.shape}")
    # label the connected components
    
    """ node_indices holds the connection label for each point. These are just the integer labels, but they are now in the same order as the sorted group_indices from arg
    
    Reshaping this means that edges are [start, stop], so [0 1] is saying that node at index 0 connects to the node at index 1 (I think?) 
    
    """

    edges = node_indices.reshape(-1, 2)
    

    
    if debug==True: print(f"edges shape: {edges.shape}")
    if debug==True: print(f"edges : {edges}")
    conn_labels = tm.graph.connected_component_labels(edges, node_count=len(grouped_rows))
    if debug==True: print(f"conn_labels: {conn_labels}")
    # conn labels correspond to the nodes. Label each edge by one of its nodes.
    edge_labels = conn_labels[edges[:, 0]]
    if debug==True: print(f"edge labels: {edge_labels}")

    """ 
    I don't quite understand this yet. The idea is to get the intersection_lines that correspond to the indices given by the edge_labels. But I can get the same first few arrays without doing this. 
    
    I simply take 
    
    #convert coordinates to [[start x, start y], [stop x, stop y]]
    linecoords = plane_line.reshape(-1,2)

    # do the same sorting
    arg = np.argsort(grouped_indices[:, 0])
    node_ind1 = grouped_indices[arg, 1]
    edges = node_ind1.reshape(-1, 2)

    # use the edges to get coordinates of connected nodes
    edge_points = linecoords[edges]
    
    
    my only thought is that my rough attempt fails when the connectivity gets higher? Not sure. 
    
    """
    unique_lbls = np.unique(conn_labels)
    split_intersection = []
    for lbl in unique_lbls:
        split_intersection.append(intersection[edge_labels == lbl, :, :])
    return split_intersection



def get_branch_connections(branch_intersections_slices, connection_distance):
    """Gets the connections between branches organized in slices

    Args:
        branch_intersections_slices (list of lists of arrays): For each slice, for each branch, array of points in that branch.
        connection_distance (float): Distance for which branches are considered connected.

    Returns:
        list of list of arrays:  For each slice, for each branch, which branches from layer below is it connected to.
    """
    branch_connections = []
    # with Parallel(n_jobs=5, backend="threading") as parallel:
    for i, separated_pts in enumerate(branch_intersections_slices):
        if i == 0:
            branch_connections.append([])
            separated_pts_below = separated_pts
            continue
        this_slice_connections = []
        for br_pts in separated_pts:
            # get how the branches in this layer connect to the branches from the previous layer. This is done by checking if the branches are neighbours (within a connection_distance away)
            this_branch_connections = []
            tree = KDTree(br_pts.reshape(-1, 2))
            branch_min_distances = []
            for k, branch_pts_below in enumerate(separated_pts_below):
                nb, dist = is_branch_nb(tree, branch_pts_below, connection_distance)
                if nb:
                    branch_min_distances.append(dist)
                    this_branch_connections.append(k)
            this_branch_connections = np.array(this_branch_connections)
            n_conn = len(this_branch_connections)

            # the previous has an issue. The two branches can be close, but not fully merge until a few layers above.
            # This is a hack to fix it. It looks if the branches that seem to connect to each other exist (within a connection distance) in the current layer.
            # If something does not work properly, this is the likely culprit.
            # There must be a better way of doing this
            if n_conn > 1:
                # Ensure that the branches that were merged do not exist in the current layer
                # to know how many duplicates, need to know how many branches in this layer are within a conn distance away
                count = 0
                for br_pts2 in separated_pts:
                    if is_branch_nb(tree, br_pts2, connection_distance)[0]:
                        count += 1
                if count > 1 and count <= n_conn:
                    # drop the extra connections
                    n_drop = n_conn - count + 1
                    keep_indx = np.argsort(branch_min_distances)[:n_drop]
                    this_branch_connections = this_branch_connections[keep_indx]
            this_slice_connections.append(this_branch_connections)
        branch_connections.append(this_slice_connections)
        separated_pts_below = separated_pts
    return branch_connections



import matplotlib.pyplot as plt


lines, levels = get_intersection_lines(1, mesh)

plane_line = lines[10]
plane_line = plane_line[:10]

split_ints = split_intersection(plane_line)

grouped_rows = tm.grouping.group_rows(plane_line.reshape(-1, 2))
    
grouped_indices = np.array(
        [[l, i] for i, ls in enumerate(grouped_rows) for l in ls]
    )
grouped_indices = grouped_indices[:, :]




linecoords = plane_line.reshape(-1,2)

arg = np.argsort(grouped_indices[:, 0])
node_ind1 = grouped_indices[arg, 1]
edges = node_ind1.reshape(-1, 2)


edge_points = linecoords[edges]
# print(edge_points)
# print(split_ints)


# edge_start = edge_points[:, 0, :]
# edge_end = edge_points[:,1,:]
# print(edge_start)
# print("\n")
# # node_ind2 = np.sort(grouped_indices, axis=1)

# plt.scatter(linecoords[:,0], linecoords[:,1])

# plt.figure()
# for n in range(edge_points.shape[0]):
#     start_points = edge_points[n, 0,:]
#     end_points = edge_points[n, 1,:]
#     plt.plot((start_points[0], end_points[0]), (start_points[1], end_points[1]))
