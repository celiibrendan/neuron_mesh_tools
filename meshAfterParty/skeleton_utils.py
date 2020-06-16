"""
These functions will help with skeletonization

"""
import numpy_utils as nu
import trimesh_utils as tu
from trimesh_utils import split_significant_pieces,split,combine_meshes,write_neuron_off,grouping_containing_mesh_indices

from soma_extraction_utils import find_soma_centroids,find_soma_centroid_containing_meshes

import numpy as np
import trimesh


def save_skeleton_cgal(surface_with_poisson_skeleton,largest_mesh_path):
    """
    surface_with_poisson_skeleton (np.array) : nx2 matrix with the nodes
    """
    first_node = surface_with_poisson_skeleton[0][0]
    end_nodes =  surface_with_poisson_skeleton[:,1]
    
    skeleton_to_write = str(len(end_nodes) + 1) + " " + str(first_node[0]) + " " +  str(first_node[1]) + " " +  str(first_node[2])
    
    for node in end_nodes:
        skeleton_to_write +=  " " + str(node[0]) + " " +  str(node[1]) + " " +  str(node[2])
    
    output_file = largest_mesh_path
    if output_file[-5:] != ".cgal":
        output_file += ".cgal"
        
    f = open(output_file,"w")
    f.write(skeleton_to_write)
    f.close()
    return 

#read in the skeleton files into an array
def read_skeleton_edges_coordinates(file_path):
    if type(file_path) == str or type(file_path) == type(Path()):
        file_path = [file_path]
    elif type(file_path) == list:
        pass
    else:
        raise Exception("file_path not a string or list")
    new_file_path = []
    for f in file_path:
        if type(f) == type(Path()):
            new_file_path.append(str(f.absolute()))
        else:
            new_file_path.append(str(f))
    file_path = new_file_path
    
    total_skeletons = []
    for fil in file_path:
        try:
            with open(fil) as f:
                bones = np.array([])
                for line in f.readlines():
                    #print(line)
                    line = (np.array(line.split()[1:], float).reshape(-1, 3))
                    #print(line[:-1])
                    #print(line[1:])

                    #print(bones.size)
                    if bones.size <= 0:
                        bones = np.stack((line[:-1],line[1:]),axis=1)
                    else:
                        bones = np.vstack((bones,(np.stack((line[:-1],line[1:]),axis=1))))
                    #print(bones)
                total_skeletons.append(np.array(bones).astype(float))
        except:
            print(f"file {fil} not found so skipping")
    
    return stack_skeletons(total_skeletons)
#     if len(total_skeletons) > 1:
#         returned_skeleton = np.vstack(total_skeletons)
#         return returned_skeleton
#     if len(total_skeletons) == 0:
#         print("There was no skeletons found for these files")
#     return np.array(total_skeletons).reshape(-1,2,3)

#read in the skeleton files into an array
def read_skeleton_verts_edges(file_path):
    with open(file_path) as f:
        bones = np.array([])
        for line in f.readlines():
            #print(line)
            line = (np.array(line.split()[1:], float).reshape(-1, 3))
            #print(line[:-1])
            #print(line[1:])

            #print(bones.size)
            if bones.size <= 0:
                bones = np.stack((line[:-1],line[1:]),axis=1)
            else:
                bones = np.vstack((bones,(np.stack((line[:-1],line[1:]),axis=1))))
            #print(bones)
    
    bones_array = np.array(bones).astype(float)
    
    #unpacks so just list of vertices
    vertices_unpacked  = bones_array.reshape(-1,3)

    #reduce the number of repeat vertices and convert to list
    unique_rows = np.unique(vertices_unpacked, axis=0)
    unique_rows_list = unique_rows.tolist()

    #assigns the number to the vertex (in the original vertex list) that corresponds to the index in the unique list
    vertices_unpacked_coefficients = np.array([unique_rows_list.index(a) for a in vertices_unpacked.tolist()])

    #reshapes the vertex list to become an edge list (just with the labels so can put into netowrkx graph)
    edges_with_coefficients =  np.array(vertices_unpacked_coefficients).reshape(-1,2)

    return unique_rows, edges_with_coefficients

def convert_skeleton_to_nodes_edges(bones_array):
    #unpacks so just list of vertices
    vertices_unpacked  = bones_array.reshape(-1,3)

    #reduce the number of repeat vertices and convert to list
    unique_rows = np.unique(vertices_unpacked, axis=0)
    unique_rows_list = unique_rows.tolist()

    #assigns the number to the vertex (in the original vertex list) that corresponds to the index in the unique list
    vertices_unpacked_coefficients = np.array([unique_rows_list.index(a) for a in vertices_unpacked.tolist()])

    #reshapes the vertex list to become an edge list (just with the labels so can put into netowrkx graph)
    edges_with_coefficients =  np.array(vertices_unpacked_coefficients).reshape(-1,2)

    return unique_rows, edges_with_coefficients


    
def calculate_skeleton_distance(my_skeleton):
    total_distance = np.sum(np.sqrt(np.sum((my_skeleton[:,0] - my_skeleton[:,1])**2,axis=1)))
    return float(total_distance)


import ipyvolume as ipv

def plot_ipv_mesh(elephant_mesh_sub,color=[1.,0.,0.,0.2]):
    mesh4 = ipv.plot_trisurf(elephant_mesh_sub.vertices[:,0],
                               elephant_mesh_sub.vertices[:,1],
                               elephant_mesh_sub.vertices[:,2],
                               triangles=elephant_mesh_sub.faces)

    mesh4.color = color
    mesh4.material.transparent = True

def graph_skeleton_and_mesh(main_mesh_verts=[],
                            main_mesh_faces=[],
                            unique_skeleton_verts_final=[],
                            edges_final=[],
                            edge_coordinates=[],
                            other_meshes=[],
                            other_meshes_colors =  [],
                            main_mesh_color = [0.,1.,0.,0.2],
                            main_skeleton_color = [0,0.,1,1],
                            main_mesh_face_coloring = [],
                           buffer=0,
                           axis_box_off=True,
                           html_path=""):
    """
    Graph the final result: 
    """

    ipv.figure(figsize=(15,15))
    main_mesh_vertices = []
    
    if (len(unique_skeleton_verts_final) > 0 and len(edges_final) > 0) or (len(edge_coordinates)>0):
        if (len(edge_coordinates)>0):
            unique_skeleton_verts_final,edges_final = convert_skeleton_to_nodes_edges(edge_coordinates)
        mesh2 = ipv.plot_trisurf(unique_skeleton_verts_final[:,0], 
                                unique_skeleton_verts_final[:,1], 
                                unique_skeleton_verts_final[:,2], 
                                lines=edges_final, color='blue')

        mesh2.color = main_skeleton_color 
        mesh2.material.transparent = True
        
        main_mesh_vertices.append(unique_skeleton_verts_final)
    
    if len(main_mesh_verts) > 0 and len(main_mesh_faces) > 0:
        if len(main_mesh_face_coloring) > 0:
            #will go through and color the faces of the main mesh if any sent
            for face_array,face_color in main_mesh_face_coloring:
                curr_mesh = main_mesh.submesh([face_array],append=True)
                plot_ipv_mesh(curr_mesh,face_color)
        else:
            main_mesh = trimesh.Trimesh(vertices=main_mesh_verts,faces=main_mesh_faces)

            mesh3 = ipv.plot_trisurf(main_mesh.vertices[:,0],
                                   main_mesh.vertices[:,1],
                                   main_mesh.vertices[:,2],
                                   triangles=main_mesh.faces)
            mesh3.color = main_mesh_color
            mesh3.material.transparent = True
            
        main_mesh_vertices.append(main_mesh_verts)
        
    for curr_mesh,curr_color in zip(other_meshes,other_meshes_colors):
        plot_ipv_mesh(curr_mesh,color=curr_color)
        main_mesh_vertices.append(curr_mesh.vertices)
    

    #create the main mesh vertices for setting the bounding box
    if len(main_mesh_vertices) == 0:
        raise Exception("No meshes or skeletons passed to the plotting funciton")
    elif len(main_mesh_vertices) == 1:
        main_mesh_vertices = main_mesh_vertices[0]
    else:
        main_mesh_vertices = np.vstack(main_mesh_vertices)
    
    volume_max = np.max(main_mesh_vertices,axis=0)
    volume_min = np.min(main_mesh_vertices,axis=0)

    ranges = volume_max - volume_min
    index = [0,1,2]
    max_index = np.argmax(ranges)
    min_limits = [0,0,0]
    max_limits = [0,0,0]


    for i in index:
        if i == max_index:
            min_limits[i] = volume_min[i] - buffer
            max_limits[i] = volume_max[i] + buffer 
            continue
        else:
            difference = ranges[max_index] - ranges[i]
            min_limits[i] = volume_min[i] - difference/2  - buffer
            max_limits[i] = volume_max[i] + difference/2 + buffer

    #ipv.xyzlim(-2, 2)
    ipv.xlim(min_limits[0],max_limits[0])
    ipv.ylim(min_limits[1],max_limits[1])
    ipv.zlim(min_limits[2],max_limits[2])
    
    
    ipv.style.set_style_light()
    if axis_box_off:
        ipv.style.axes_off()
        ipv.style.box_off()
    else:
        ipv.style.axes_on()
        ipv.style.box_on()
    ipv.show()
    
    if html_path != "":
        ipv.pylab.save(html_path)
        


""" ------------------- Mesh subtraction ------------------------------------"""
import numpy as np
#make sure pip3 install trimesh --upgrade so can have slice
import trimesh 
import matplotlib.pyplot as plt
import ipyvolume as ipv
import calcification_Module as cm
from pathlib import Path
import time
from tqdm.notebook import tqdm
import skeleton_utils as sk

#  Utility functions
angle = np.pi/2
rotation_matrix = np.array([[np.cos(angle),-np.sin(angle),0],
                            [np.sin(angle),np.cos(angle),0],
                            [0,0,1]
                           ])

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q
def change_basis_matrix(v):
    """
    This just gives change of basis matrix for a basis 
    that has the vector v as its 3rd basis vector
    and the other 2 vectors orthogonal to v 
    (but not necessarily orthogonal to each other)
    *** not make an orthonormal basis ***
    
    -- changed so now pass the non-orthogonal components
    to the QR decomposition to get them as orthogonal
    
    """
    a,b,c = v
    #print(f"a,b,c = {(a,b,c)}")
    if np.abs(c) > 0.00001:
        v_z = v/np.linalg.norm(v)
        v_x = np.array([1,0,-a/c])
        #v_x = v_x/np.linalg.norm(v_x)
        v_y = np.array([0,1,-b/c])
        #v_y = v_y/np.linalg.norm(v_y)
        v_x, v_y = gram_schmidt_columns(np.vstack([v_x,v_y]).T).T
        return np.vstack([v_x,v_y,v_z])
    else:
        #print("Z coeffienct too small")
        #v_z = v
        v[2] = 0
        #print(f"before norm v_z = {v}")
        v_z = v/np.linalg.norm(v)
        #print(f"after norm v_z = {v_z}")
        
        v_x = np.array([0,0,1])
        v_y = rotation_matrix@v_z
        
    return np.vstack([v_x,v_y,v_z])

def mesh_subtraction_by_skeleton(main_mesh,edges,
                                 buffer=0.01,
                                bbox_ratio=1.2,
                                 distance_threshold=500,
                             significance_threshold=500,
                                print_flag=False):
    """
    Purpose: Will return significant mesh pieces that are
    not already accounteed for by the skeleton
    
    Example of how to run
    
    main_mesh_path = Path("./Dustin/Dustin.off")
    main_mesh = trimesh.load_mesh(str(main_mesh_path.absolute()))
    skeleton_path = main_mesh_path.parents[0] / Path(main_mesh_path.stem + "_skeleton.cgal")
    edges = sk.read_skeleton_edges_coordinates(str(skeleton_path.absolute()))

    # turn this into nodes and edges
    main_mesh_nodes, main_mesh_edges = sk.read_skeleton_verts_edges(str(skeleton_path.absolute()))
    sk.graph_skeleton_and_mesh(
                main_mesh_verts=main_mesh.vertices,
                main_mesh_faces=main_mesh.faces,
                unique_skeleton_verts_final = main_mesh_nodes,
                edges_final=main_mesh_edges,
                buffer = 0
                              )
                              
    leftover_pieces =  mesh_subtraction_by_skeleton(main_mesh,edges,
                                 buffer=0.01,
                                bbox_ratio=1.2,
                                 distance_threshold=500,
                             significance_threshold=500,
                                print_flag=False)
                                
    # Visualize the results: 
    pieces_mesh = trimesh.Trimesh(vertices=np.array([]),
                                 faces=np.array([]))

    for l in leftover_pieces:
        pieces_mesh += l

    sk.graph_skeleton_and_mesh(
                main_mesh_verts=pieces_mesh.vertices,
                main_mesh_faces=pieces_mesh.faces,
                unique_skeleton_verts_final = main_mesh_nodes,
                edges_final=main_mesh_edges,
                buffer = 0
                              )
    
    """
    
    skeleton_nodes = edges.reshape(-1,3)
    skeleton_bounding_corners = np.vstack([np.max(skeleton_nodes,axis=0),
               np.min(skeleton_nodes,axis=0)])
    
    main_mesh_bbox_restricted, faces_bbox_inclusion = tu.bbox_mesh_restrcition(main_mesh,
                                                                        skeleton_bounding_corners,
                                                                        bbox_ratio)

    if type(main_mesh_bbox_restricted) == type(trimesh.Trimesh()):
        print(f"Inside mesh subtraction, len(main_mesh_bbox_restricted.faces) = {len(main_mesh_bbox_restricted.faces)}")
    else:
        print("***** Bounding Box Restricted Mesh is empty ****")
        main_mesh_bbox_restricted = main_mesh
        faces_bbox_inclusion = np.arange(0,len(main_mesh.faces))
    
    start_time = time.time()

    #face_subtract_color = []
    face_subtract_indices = []

    distance_threshold = 2000
    
    edge_loop_print=False
    for i,ex_edge in tqdm(enumerate(edges)):
        #print("\n------ New loop ------")
        #print(ex_edge)
        
        # ----------- creating edge and checking distance ----- #
        loop_start = time.time()
        
        edge_line = ex_edge[1] - ex_edge[0]
        sum_threshold = 0.001
        if np.sum(np.abs(edge_line)) < sum_threshold:
            if edge_loop_print:
                print(f"edge number {i}, {ex_edge}: has sum less than {sum_threshold} so skipping")
            continue
        if edge_loop_print:
            print(f"Checking Edge Distance = {time.time()-loop_start}")
        
        loop_start = time.time()
        cob_edge = change_basis_matrix(edge_line)
        if edge_loop_print:
            print(f"Change of Basis Matrix calculation = {time.time()-loop_start}")

        loop_start - time.time()
        #get the limits of the example edge itself that should be cutoff
        slice_range = np.sort((cob_edge@ex_edge.T)[2,:])

        # adding the buffer to the slice range
        slice_range_buffer = slice_range + np.array([-buffer,buffer])
        if edge_loop_print:
            print(f"Calculate slice= {time.time()-loop_start}")
        loop_start = time.time()

        # generate face midpoints from the triangles
        #face_midpoints = np.mean(main_mesh_bbox_restricted.vertices[main_mesh_bbox_restricted.faces],axis=1) # Old way
        face_midpoints = main_mesh_bbox_restricted.triangles_center
        if edge_loop_print:
            print(f"Face midpoints= {time.time()-loop_start}")
        loop_start = time.time()
        #get the face midpoints that fall within the slice (by lookig at the z component)
        fac_midpoints_trans = cob_edge@face_midpoints.T
        if edge_loop_print:
            print(f"Face midpoints transform= {time.time()-loop_start}")
        loop_start = time.time()
        edge_midpoint = np.mean(ex_edge,axis=0)
        if edge_loop_print:
            print(f"edge midpoint= {time.time()-loop_start}")
        loop_start = time.time()
        
        slice_mask_pre_distance = ((fac_midpoints_trans[2,:]>slice_range_buffer[0]) & 
                      (fac_midpoints_trans[2,:]<slice_range_buffer[1]))

        if edge_loop_print:
            print(f"Applying slice restriction = {time.time()-loop_start}")
        loop_start = time.time()
        # apply the distance threshold to the slice mask
        #raise Exception("Add in part for distance threshold here")
        distance_check = np.linalg.norm(face_midpoints[:,:2] - edge_midpoint[:2],axis=1) < distance_threshold
        slice_mask = slice_mask_pre_distance & distance_check
        
        if edge_loop_print:
            print(f"Applying distance restriction= {time.time()-loop_start}")
        loop_start = time.time()


        face_list = np.arange(0,len(main_mesh_bbox_restricted.faces))[slice_mask]

        #get the submesh of valid faces within the slice
        if len(face_list)>0:
            main_mesh_sub = main_mesh_bbox_restricted.submesh([face_list],append=True)
        else:
            main_mesh_sub = []
        
        

        if type(main_mesh_sub) != type(trimesh.Trimesh()):
            if edge_loop_print:
                print(f"THERE WERE NO FACES THAT FIT THE DISTANCE ({distance_threshold}) and Z transform requirements")
                print("So just skipping this edge")
            continue

        if edge_loop_print:
            print(f"getting submesh= {time.time()-loop_start}")
        loop_start = time.time()
        
        #get all disconnected mesh pieces of the submesh and the face indices for lookup later
        sub_components,sub_components_face_indexes = sk.split(main_mesh_sub,only_watertight=False)
        if type(sub_components) != type(np.array([])) and type(sub_components) != list:
            print(f"meshes = {sub_components}, with type = {type(sub_components)}")
            if type(sub_components) == type(trimesh.Trimesh()) :
                sub_components = [sub_components]
            else:
                raise Exception("The sub_components were not an array, list or trimesh")
        
        if edge_loop_print:
            print(f"splitting the mesh= {time.time()-loop_start}")
        loop_start = time.time()

        #getting the indices of the submeshes whose bounding box contain the edge 
        contains_points_results = np.array([s_comp.bounding_box.contains(ex_edge.reshape(-1,3)) for s_comp in sub_components])
        containing_indices = (np.arange(0,len(sub_components)))[np.sum(contains_points_results,axis=1) >= len(ex_edge)]
        
        if edge_loop_print:
            print(f"containing indices= {time.time()-loop_start}")
        loop_start = time.time()

        if len(containing_indices) != 1: 
            if edge_loop_print:
                print(f"--> Not exactly one containing mesh: {containing_indices}")
            if len(containing_indices) > 1:
                sub_components_inner = sub_components[containing_indices]
                sub_components_face_indexes_inner = sub_components_face_indexes[containing_indices]
            else:
                sub_components_inner = sub_components
                sub_components_face_indexes_inner = sub_components_face_indexes

            #get the center of the edge
            edge_center = np.mean(ex_edge,axis=0)
            #print(f"edge_center = {edge_center}")

            #find the distance between eacch bbox center and the edge center
            bbox_centers = [np.mean(k.bounds,axis=0) for k in sub_components_inner]
            #print(f"bbox_centers = {bbox_centers}")
            closest_bbox = np.argmin([np.linalg.norm(edge_center-b_center) for b_center in bbox_centers])
            #print(f"bbox_distance = {closest_bbox}")
            edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes_inner[closest_bbox]]]
            if edge_loop_print:
                print(f"finding closest box when 0 or 2 or more containing boxes= {time.time()-loop_start}")
            loop_start = time.time()
        else:# when only one viable submesh piece and just using that sole index
            #print(f"only one viable submesh piece so using index only number in: {containing_indices}")
            
            edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes[containing_indices[0]]]]
            if edge_loop_print:
                print(f"only 1 containig face getting the edge_skeleton_faces= {time.time()-loop_start}")
            loop_start = time.time()

        if len(edge_skeleton_faces) < 0:
            print(f"****** Warning the edge index {i}: had no faces in the edge_skeleton_faces*******")
        face_subtract_indices.append(edge_skeleton_faces)
        if edge_loop_print:
                print(f"check and append for face= {time.time()-loop_start}")
        #face_subtract_color.append(viable_colors[i%len(viable_colors)])
        
    print(f"Total Mesh subtraction time = {np.round(time.time() - start_time,4)}")
    
    if len(face_subtract_indices)>0:
        all_removed_faces = np.concatenate(face_subtract_indices)

        unique_removed_faces = set(all_removed_faces)

        faces_to_keep = set(np.arange(0,len(main_mesh.faces))).difference(unique_removed_faces)
        new_submesh = main_mesh.submesh([list(faces_to_keep)],only_watertight=False,append=True)
    else:
        new_submesh = main_mesh
    
    significant_pieces = split_significant_pieces(new_submesh,
                                                         significance_threshold,
                                                         print_flag=False)

    return significant_pieces

""" ------------------- End of Mesh Subtraction ------------------------------------"""



""" ----------Start of Surface Skeeltonization -- """

import networkx as nx
import time 
import numpy as np
import trimesh
import random


# # Older version that was not working properly
# def generate_surface_skeleton(vertices,
#                               faces, 
#                               surface_samples=1000,
#                           print_flag=False):
    
#     #return surface_with_poisson_skeleton,path_length
    
#     mesh = trimesh.Trimesh(vertices=vertices,
#                                   faces = faces,
#                            )


#     start_time = time.time()

#     ga = nx.from_edgelist(mesh.edges)

#     if surface_samples<len(vertices):
#         k = surface_samples
#     else:
#         k = len(vertices)
#     sampled_nodes = random.sample(ga.nodes, k)


#     lp_end_list = []
#     lp_magnitude_list = []

#     for s in sampled_nodes: 
#         sp_dict = nx.single_source_shortest_path_length(ga,s)

#         list_keys = list(sp_dict.keys())
#         longest_path_node = list_keys[len(list_keys)-1]
#         longest_path_magnitude = sp_dict[longest_path_node]


#         lp_end_list.append(longest_path_node)
#         lp_magnitude_list.append(longest_path_magnitude)

#     #construct skeleton from shortest path
#     final_start = sampled_nodes[np.argmax(lp_magnitude_list)]
#     final_end = sampled_nodes[np.argmax(lp_end_list)]

#     node_list = nx.shortest_path(ga,final_start,final_end)
#     if len(node_list) < 2:
#         print("node_list len < 2 so returning empty list")
#         return np.array([])
#     #print("node_list = " + str(node_list))

#     final_skeleton = mesh.vertices[np.vstack([node_list[:-1],node_list[1:]]).T]
#     if print_flag:
#         print(f"   Final Time for surface skeleton with sample size = {k} = {time.time() - start_time}")

#     return final_skeleton


def generate_surface_skeleton(vertices,
                              faces, 
                              surface_samples=1000,
                              n_surface_downsampling=0,
                          print_flag=False):
    
    #return surface_with_poisson_skeleton,path_length
    
    mesh = trimesh.Trimesh(vertices=vertices,
                                  faces = faces,
                           )


    start_time = time.time()

    ga = nx.from_edgelist(mesh.edges)

    if surface_samples<len(vertices):
        sampled_nodes = np.random.choice(len(vertices),surface_samples , replace=False)
    else:
        if print_flag:
            print("Number of surface samples exceeded number of vertices, using len(vertices)")
        sampled_nodes = np.arange(0,len(vertices))
        
    lp_end_list = []
    lp_magnitude_list = []

    for s in sampled_nodes: 
        #gives a dictionary where the key is the end node and the value is the number of
        # edges on the shortest path
        sp_dict = nx.single_source_shortest_path_length(ga,s)

        #
        list_keys = list(sp_dict.keys())
        
        #gets the end node that would make the longest shortest path 
        longest_path_node = list_keys[-1]
        
        #get the number of edges for the path
        longest_path_magnitude = sp_dict[longest_path_node]


        #add the ending node and the magnitude of it to lists
        lp_end_list.append(longest_path_node)
        lp_magnitude_list.append(longest_path_magnitude)

    lp_end_list = np.array(lp_end_list)
    #construct skeleton from shortest path
    max_index = np.argmax(lp_magnitude_list)
    final_start = sampled_nodes[max_index]
    final_end = lp_end_list[max_index]

    node_list = nx.shortest_path(ga,final_start,final_end)
    if len(node_list) < 2:
        print("node_list len < 2 so returning empty list")
        return np.array([])
    #print("node_list = " + str(node_list))

    final_skeleton = mesh.vertices[np.vstack([node_list[:-1],node_list[1:]]).T]
    if print_flag:
        print(f"   Final Time for surface skeleton with sample size = {k} = {time.time() - start_time}")
        
    for i in range(n_surface_downsampling):
        final_skeleton = downsample_skeleton(final_skeleton)

    return final_skeleton


def downsample_skeleton(current_skeleton):
    #print("current_skeleton = " + str(current_skeleton.shape))
    """
    Downsamples the skeleton by 50% number of edges
    """
    extra_segment = []
    if current_skeleton.shape[0] % 2 != 0:
        extra_segment = np.array([current_skeleton[0]])
        current_skeleton = current_skeleton[1:]
        #print("extra_segment = " + str(extra_segment))
        #print("extra_segment.shape = " + str(extra_segment.shape))
    else:
        #print("extra_segment = " + str(extra_segment))
        pass

    even_indices = [k for k in range(0,current_skeleton.shape[0]) if k%2 == 0]
    odd_indices = [k for k in range(0,current_skeleton.shape[0]) if k%2 == 1]
    even_verts = current_skeleton[even_indices,0,:]
    odd_verts = current_skeleton[odd_indices,1,:]

    downsampled_skeleton = np.hstack([even_verts,odd_verts]).reshape(even_verts.shape[0],2,3)
    #print("dowsampled_skeleton.shape = " + str(downsampled_skeleton.shape))
    if len(extra_segment) > 0:
        #print("downsampled_skeleton = " + str(downsampled_skeleton.shape))
        final_downsampled_skeleton = np.vstack([extra_segment,downsampled_skeleton])
    else:
        final_downsampled_skeleton = downsampled_skeleton
    return final_downsampled_skeleton


# ----- Stitching Algorithm ----- #
import networkx as nx

from pykdtree.kdtree import KDTree
from tqdm.notebook import tqdm

import scipy
def stitch_skeleton(
                                          staring_edges,
                                          max_stitch_distance=18000,
                                          stitch_print = False,
                                          main_mesh = []
                                        ):

    stitched_time = time.time()

    stitch_start = time.time()

    all_skeleton_vertices = staring_edges.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    edges_with_coefficients = indices.reshape(-1,2)

    if stitch_print:
        print(f"Getting the unique rows and indices= {time.time()-stitch_start}")
    stitch_start = time.time()

    #create the graph from the edges
    B = nx.Graph()
    B.add_nodes_from([(x,{"coordinates":y}) for x,y in enumerate(unique_rows)])
    B.add_edges_from(edges_with_coefficients)

    if stitch_print:
        print(f"Putting edges into networkx graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix

    UG = B.to_undirected()

    if stitch_print:
        print(f"Making undirected graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    #get all of the coordinates

    print("len_subgraphs AT BEGINNING of the loop")
    counter = 0
    print_flag = True

    n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=nx.adjacency_matrix(UG), directed=False, return_labels=True)
    #print(f"At beginning n_components = {n_components}, unique labels = {np.unique(labels)}")
    
    subgraph_components = np.where(labels==0)[0]
    outside_components = np.where(labels !=0)[0]

    for j in tqdm(range(n_components)):
        counter+= 1
        if stitch_print:
            print(f"Starting Loop {counter}")
        start_time = time.time()
        """
        1) Get the indexes of the subgraph
        2) Build a KDTree from those not in the subgraph (save the vertices of these)
        3) Query against the nodes in the subgraph  and get the smallest distance
        4) Create this new edge

        """
        stitch_time = time.time()
        #1) Get the indexes of the subgraph
        #n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=nx.adjacency_matrix(UG), directed=False, return_labels=True)
        if stitch_print:
            print(f"Finding Number of Connected Components= {time.time()-stitch_start}")
        stitch_start = time.time()

        subgraph_components = np.where(labels==0)[0]

        if stitch_print:
            print(f"Faces belonging to largest component= {time.time()-stitch_start}")
        stitch_start = time.time()
        #print("subgraph_components = " + str(subgraph_components))
        if len(subgraph_components) == len(UG.nodes):
            print("all graph is one component!")
            #print(f"unique labels = {np.unique(labels)}")
            break

        if stitch_print:
            print(f"n_components = {n_components}")

        outside_components = np.where(labels !=0)[0]

        if stitch_print:
            print(f"finding faces not in largest component= {time.time()-stitch_start}")
        stitch_start = time.time()
        #print("outside_components = " + str(outside_components))

        #2) Build a KDTree from those not in the subgraph (save the vertices of these)
        mesh_tree = KDTree(unique_rows[outside_components])
        if stitch_print:
            print(f"Building KDTree= {time.time()-stitch_start}")
        stitch_start = time.time()


        #3) Query against the nodes in the subgraph  and get the smallest distance
        """
        Conclusion:
        Distance is of the size of the parts that are in the KDTree
        The closest nodes represent those that were queryed

        """
        distances,closest_node = mesh_tree.query(unique_rows[subgraph_components])
        if stitch_print:
            print(f"Mesh Tree query= {time.time()-stitch_start}")
        stitch_start = time.time()
        min_index = np.argmin(distances)
        
        #check if the distance is too far away 
        if distances[min_index] > max_stitch_distance:
            print(f"**** The distance exceeded max stitch distance of {max_stitch_distance}"
                   f" and still had {n_components} left\n"
                  f"   Actual distance was {distances[min_index]} ")
        

        if stitch_print:
            print(f"Finding closest distance= {time.time()-stitch_start}")
        stitch_start = time.time()


        closest_outside_node = outside_components[closest_node[min_index]]
        closest_subgraph_node = subgraph_components[min_index]

        if stitch_print:
            print(f"Getting nodes to be paired up= {time.time()-stitch_start}")
        stitch_start = time.time()

        
        
        #get the edge distance of edge about to create:

    #         graph_coordinates=nx.get_node_attributes(UG,'coordinates')
    #         prospective_edge_length = np.linalg.norm(np.array(graph_coordinates[closest_outside_node])-np.array(graph_coordinates[closest_subgraph_node]))
    #         print(f"Edge distance going to create = {prospective_edge_length}")

        #4) Create this new edge
        UG.add_edge(closest_subgraph_node,closest_outside_node)

        #get the label of the closest outside node 
        closest_outside_label = labels[closest_outside_node]

        #get all of the nodes with that same label
        new_subgraph_components = np.where(labels==closest_outside_label)[0]

        #relabel the nodes so now apart of the largest component
        labels[new_subgraph_components] = 0

        #move the newly relabeled nodes out of the outside components into the subgraph components
        ## --- SKIP THIS ADDITION FOR RIGHT NOW -- #


        if stitch_print:
            print(f"Adding Edge = {time.time()-stitch_start}")
        stitch_start = time.time()

        n_components -= 1

        if stitch_print:
            print(f"Total Time for loop = {time.time() - start_time}")


    # get the largest subgraph!!! in case have missing pieces

    #add all the new edges to the 

#     total_coord = nx.get_node_attributes(UG,'coordinates')
#     current_coordinates = np.array(list(total_coord.values()))

    current_coordinates = unique_rows
    
    try:
        total_edges_stitched = current_coordinates[np.array(list(UG.edges())).reshape(-1,2)]
    except:
        print("getting the total edges stitched didn't work")
        print(f"current_coordinates = {current_coordinates}")
        print(f"UG.edges() = {list(UG.edges())} with type = {type(list(UG.edges()))}")
        print(f"np.array(UG.edges()) = {np.array(list(UG.edges()))}")
        print(f"np.array(UG.edges()).reshape(-1,2) = {np.array(list(UG.edges())).reshape(-1,2)}")
        
        raise Exception(" total_edges_stitched not calculated")
        #print("returning ")
        #total_edges_stitched
    

    print(f"Total time for skeleton stitching = {time.time() - stitched_time}")
    
    return total_edges_stitched


def stack_skeletons(sk_list):
    list_of_skeletons = [np.array(k).reshape(-1,2,3) for k in sk_list if len(k)>0]
    if len(list_of_skeletons) == 0:
        print("No skeletons to stack so returning empty list")
        return []
    elif len(list_of_skeletons) == 1:
        print("only one skeleton so no stacking needed")
        return np.array(list_of_skeletons).reshape(-1,2,3)
    else:
        return (np.vstack(list_of_skeletons)).reshape(-1,2,3)

#------------ The actual skeletonization from mesh contraction----------- #
from calcification_param_Module import calcification_param
def calcification(
                    location_with_filename,
                    max_triangle_angle =1.91986,
                    quality_speed_tradeoff=0.1,
                    medially_centered_speed_tradeoff=0.2,
                    area_variation_factor=0.0001,
                    max_iterations=500,
                    is_medially_centered=True,
                    min_edge_length = 0,
                    edge_length_multiplier = 0.002,
                    print_parameters=True
                ):
    
    if type(location_with_filename) == type(Path()):
        location_with_filename = str(location_with_filename.absolute())
    
    if location_with_filename[-4:] == ".off":
        location_with_filename = location_with_filename[:-4]
    
    #print(f"location_with_filename = {location_with_filename}")
    
    return_value = calcification_param(
        location_with_filename,
        max_triangle_angle,
        quality_speed_tradeoff,
        medially_centered_speed_tradeoff,
        area_variation_factor,
        max_iterations,
        is_medially_centered,
        min_edge_length,
        edge_length_multiplier,
        print_parameters
    )
    
    return return_value,location_with_filename+"_skeleton.cgal"


# ---------- Does the cleaning of the skeleton -------------- #
def convert_skeleton_to_graph(staring_edges,
                             stitch_print=False):
    stitch_start = time.time()

    all_skeleton_vertices = staring_edges.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    edges_with_coefficients = indices.reshape(-1,2)

    if stitch_print:
        print(f"Getting the unique rows and indices= {time.time()-stitch_start}")
    stitch_start = time.time()

    #create the graph from the edges
    B = nx.Graph()
    B.add_nodes_from([(x,{"coordinates":y}) for x,y in enumerate(unique_rows)])
    B.add_edges_from(edges_with_coefficients)

    if stitch_print:
        print(f"Putting edges into networkx graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix

    UG = B.to_undirected()

    if stitch_print:
        print(f"Making undirected graph= {time.time()-stitch_start}")
    stitch_start = time.time()
    
    return UG



def get_skeleton_from_graph(UG):
    UG = nx.convert_node_labels_to_integers(UG)
    total_coord = nx.get_node_attributes(UG,'coordinates')
    current_coordinates = np.array(list(total_coord.values()))
    
    try:
        total_edges_stitched = current_coordinates[np.array(list(UG.edges())).reshape(-1,2)]
    except:
        print("getting the total edges stitched didn't work")
        print(f"current_coordinates = {current_coordinates}")
        print(f"UG.edges() = {list(UG.edges())} with type = {type(list(UG.edges()))}")
        print(f"np.array(UG.edges()) = {np.array(list(UG.edges()))}")
        print(f"np.array(UG.edges()).reshape(-1,2) = {np.array(list(UG.edges())).reshape(-1,2)}")
        
        raise Exception(" total_edges_stitched not calculated")
        
    return total_edges_stitched

def list_len_measure(curr_list,G):
    return len(curr_list)

def skeletal_distance(curr_list,G,coordinates_dict):
    
    #clean_time = time.time()
    #coordinates_dict = nx.get_node_attributes(G,'coordinates')
    #print(f"Extracting attributes = {time.time() - clean_time}")
    #clean_time = time.time()
    coor = [coordinates_dict[k] for k in curr_list]
    #print(f"Reading dict = {time.time() - clean_time}")
    #clean_time = time.time()
    norm_values =  [np.linalg.norm(coor[i] - coor[i-1]) for i in range(1,len(coor))]
    #print(f"Calculating norms = {time.time() - clean_time}")
    #print(f"norm_values = {norm_values}")
    return np.sum(norm_values)
    
def clean_skeleton(G,
                   distance_func,
                  min_distance_to_junction = 3,
                  return_skeleton=True,
                  print_flag=False):
    """
    Example of how to use: 
    
    Simple Example:  
    def distance_measure_func(path,G=None):
    #print("new thing")
    return len(path)

    new_G = clean_skeleton(G,distance_measure_func,return_skeleton=False)
    nx.draw(new_G,with_labels=True)
    
    More complicated example:
    
    import skeleton_utils as sk
    from importlib import reload
    sk = reload(sk)

    from pathlib import Path
    test_skeleton = Path("./Dustin_vp6/Dustin_soma_0_branch_0_0_skeleton.cgal")
    if not test_skeleton.exists():
        print(str(test_skeleton)[:-14])
        file_of_skeleton = sk.calcification(str(test_skeleton.absolute())[:-14])
    else:
        file_of_skeleton = test_skeleton

    # import the skeleton
    test_sk = sk.read_skeleton_edges_coordinates(test_skeleton)
    import trimesh
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=test_sk)

    # clean the skeleton and then visualize
    import time
    clean_time = time.time()
    cleaned_skeleton = clean_skeleton(test_sk,
                        distance_func=skeletal_distance,
                  min_distance_to_junction=10000,
                  return_skeleton=True,
                  print_flag=True)
    print(f"Total time for skeleton clean {time.time() - clean_time}")

    # see what skeleton looks like now
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=cleaned_skeleton)
                              
                              
    --------------- end of example -----------------
    """
    
    
    def end_node_path_to_junciton(curr_G,end_node):
        curr_node = end_node
        node_list = [curr_node]
        for i in range(len(curr_G)):
            neighbors = list(curr_G[curr_node])
            if len(neighbors) <= 2:
                curr_node = [k for k in neighbors if k not in node_list][0]
                node_list.append(curr_node)
                #print(f"node_list = {node_list}")
            else:
                break
        return node_list

    print(f"Using Distance measure {distance_func.__name__}")
    
    
    if type(G) != type(nx.Graph()):
        G = convert_skeleton_to_graph(G)
        
    kwargs = dict()
    kwargs["coordinates_dict"] = nx.get_node_attributes(G,'coordinates')
    
    
    end_nodes = np.array([k for k,n in dict(G.degree()).items() if n == 1])
    #print(f"len(end_nodes) = {len(end_nodes)}")
    #clean_time = time.time()
    paths_to_j = [end_node_path_to_junciton(G,n) for n in end_nodes]
    #print(f"Total time for node path to junction = {time.time() - clean_time}")
    #clean_time = time.time()
    end_nodes_dist_to_j = np.array([distance_func(n,G,**kwargs) for n in paths_to_j])
    #print(f"Calculating distances = {time.time() - clean_time}")
    #clean_time = time.time()
    
    end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
    end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]
    
    current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
    #print(f"Ordering the nodes = {time.time() - clean_time}")
    clean_time = time.time()
    if print_flag:
        print(f"total end_nodes = {end_nodes}")
    #current_end_node = ordered_end_nodes[0]
    paths_removed = 0
    
    for i in tqdm(range(len(end_nodes))):
        current_path_to_junction = end_node_path_to_junciton(G,current_end_node)
        if print_flag:
            #print(f"ordered_end_nodes = {ordered_end_nodes}")
            print(f"\n\ncurrent_end_node = {current_end_node}")
            print(f"current_path_to_junction = {current_path_to_junction}")
        if distance_func(current_path_to_junction,G,**kwargs) <min_distance_to_junction:
            if print_flag:
                print(f"the current distance that was below was {distance_func(current_path_to_junction,G,**kwargs)}")
            #remove the nodes
            paths_removed += 1
            G.remove_nodes_from(current_path_to_junction[:-1])
            end_nodes = end_nodes[end_nodes != current_end_node]
            end_nodes_dist_to_j = np.array([distance_func(end_node_path_to_junciton(G,n),G,**kwargs) for n in end_nodes])
            
            end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
            end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]
    
            if len(end_nodes_dist_to_j)<= 0:
                break
            current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
#             if print_flag:
#                 print(f"   insdie if statement ordered_end_nodes = {ordered_end_nodes}")
                
            #current_end_node = ordered_end_nodes[0]
            
        else:
            break
    if print_flag:
        print(f"Done cleaning networkx graph with {paths_removed} paths removed")
    if return_skeleton:
        return get_skeleton_from_graph(G)
    else:
        return G
    
    
# ---------------------- Full Skeletonization Function --------------------- #
from pykdtree.kdtree import KDTree
import time
import trimesh
import numpy as np
from pathlib import Path

import time
import os
import pathlib

from tqdm.notebook import tqdm

import meshlab
from importlib import reload
meshlab = reload(meshlab)
from meshlab import Decimator , Poisson
import skeleton_utils as sk

from shutil import rmtree
from pathlib import Path

import soma_extraction_utils as soma_utils
from pathlib import Path
import trimesh

def load_somas(segment_id,main_mesh_total,
              soma_path):
    soma_path = str(soma_path)
    try:
        current_soma = trimesh.load_mesh(str(soma_path))
        return [current_soma]
    except:
        print("No Soma currently available so must compute own")
        (total_soma_list, 
             run_time, 
             total_soma_list_sdf) = soma_utils.extract_soma_center(
                                segment_id,
                                main_mesh_total.vertices,
                                main_mesh_total.faces,
                                outer_decimation_ratio= 0.25,
                                large_mesh_threshold = 60000,
                                large_mesh_threshold_inner = 40000,
                                soma_width_threshold = 0.32,
                                soma_size_threshold = 20000,
                               inner_decimation_ratio = 0.25,
                               volume_mulitplier=7,
                               side_length_ratio_threshold=3,
                                soma_size_threshold_max=192000,
                                delete_files=True
            )
        
        # save the soma
        print(f"Found {len(total_soma_list)} somas")
        soma_mesh = combine_meshes(total_soma_list)
        soma_mesh.export(soma_path)
        
        return total_soma_list
    else:
        return []
    
from soma_extraction_utils import subtract_soma


def skeletonize_connected_branch(current_mesh,
                        output_folder="./temp",
                        delete_temp_files=True,
                        name="None",
                        surface_reconstruction_size=50,
                        n_surface_downsampling = 1,
                        n_surface_samples=1000,
                        skeleton_print=False,
                        mesh_subtraction_distance_threshold=3000,
                        mesh_subtraction_buffer=50,
                        max_stitch_distance = 18000,
                        current_min_edge = 200
                        ):
    """
    Purpose: To take a mesh and construct a full skeleton of it
    (Assuming the Soma is already extracted)
    
    1) Poisson Surface Reconstruction
    2) CGAL skeletonization of all signfiicant pieces 
        (if above certain size ! threshold) 
                --> if not skip straight to surface skeletonization
    3) Using CGAL skeleton, find the leftover mesh not skeletonized
    4) Do surface reconstruction on the parts that are left over
    - with some downsampling
    5) Stitch the skeleton 
    """
    
    #check that the mesh is all one piece
    current_mesh_splits = split_significant_pieces(current_mesh,
                               significance_threshold=1)
    if len(current_mesh_splits) > 1:
        raise Exception(f"The mesh passed has {len(current_mesh_splits)} pieces")

    # check the size of the branch and if small enough then just do
    # Surface Skeletonization
    if len(current_mesh.faces) < surface_reconstruction_size:
        #do a surface skeletonization
        print("Doing skeleton surface reconstruction")
        surf_sk = generate_surface_skeleton(current_mesh.vertices,
                                    current_mesh.faces,
                                    surface_samples=n_surface_samples,
                                             n_surface_downsampling=n_surface_downsampling )
        return surf_sk
    else:
    
        #if can't simply do a surface skeletonization then 
        #use cgal method that requires temp folder

        if type(output_folder) != type(Path()):
            output_folder = Path(str(output_folder))
            output_folder.mkdir(parents=True,exist_ok=True)
            
        # CGAL Step 1: Do Poisson Surface Reconstruction
        Poisson_obj = Poisson(output_folder,overwrite=True)
        

        skeleton_start = time.time()
        print("     Starting Screened Poisson")
        new_mesh,output_subprocess_obj = Poisson_obj(   
                                    vertices=current_mesh.vertices,
                                     faces=current_mesh.faces,
                                    mesh_filename=name + ".off",
                                     return_mesh=True,
                                     delete_temp_files=False,
                                    )
        print(f"-----Time for Screened Poisson= {time.time()-skeleton_start}")
            
        #2) Filter away for largest_poisson_piece:
        mesh_pieces = split_significant_pieces(new_mesh,
                                            significance_threshold=surface_reconstruction_size)
        
        if skeleton_print:
            print(f"Signifiant mesh pieces of {surface_reconstruction_size} size "
                 f"after poisson = {len(mesh_pieces)}")
        skeleton_ready_for_stitching = np.array([])
        skeleton_files = [] # to be erased later on if need be
        if len(mesh_pieces) <= 0:
            if skeleton_print:
                print("No signficant skeleton pieces so just doing surface skeletonization")
            # do surface skeletonization on all of the pieces
            surface_mesh_pieces = split_significant_pieces(new_mesh,
                                            significance_threshold=2)
            
            #get the skeletons for all those pieces
            current_mesh_skeleton_list = [
                generate_surface_skeleton(p.vertices,
                                    p.faces,
                                    surface_samples=n_surface_samples,
                                    n_surface_downsampling=n_surface_downsampling )
                for p in surface_mesh_pieces
            ]
            
            skeleton_ready_for_stitching = stack_skeletons(current_mesh_skeleton_list)
            
            #will stitch them together later
        else: #if there are parts that can do the cgal skeletonization
            skeleton_start = time.time()
            print("     Starting Calcification")
            for zz,piece in enumerate(mesh_pieces):
                current_mesh_path = output_folder / f"{name}_{zz}"
                
                written_path = write_neuron_off(piece,current_mesh_path)
                
                #print(f"Path sending to calcification = {written_path[:-4]}")
                returned_value, sk_file_name = calcification(written_path,
                                                               min_edge_length = current_min_edge)
                #print(f"Time for skeletonizatin = {time.time() - skeleton_start}")
                skeleton_files.append(sk_file_name)
                
            if skeleton_print:
                print(f"-----Time for Running Calcification = {time.time()-skeleton_start}")
            
            #collect the skeletons and subtract from the mesh
            
            significant_poisson_skeleton = read_skeleton_edges_coordinates(skeleton_files)
            
            if len(significant_poisson_skeleton) > 0:
                boolean_significance_threshold=5

                print(f"Before mesh subtraction number of skeleton edges = {significant_poisson_skeleton.shape[0]+1}")
                mesh_pieces_leftover =  mesh_subtraction_by_skeleton(current_mesh,
                                                            significant_poisson_skeleton,
                                                            buffer=mesh_subtraction_buffer,
                                                            bbox_ratio=1.2,
                                                            distance_threshold=significant_poisson_skeleton,
                                                            significance_threshold=boolean_significance_threshold,
                                                            print_flag=False
                                                           )

                # *****adding another significance threshold*****
                leftover_meshes_sig = [k for k in mesh_pieces_leftover if len(k.faces) > 50]
                leftover_meshes = combine_meshes(leftover_meshes_sig)
            else:
                print("No recorded skeleton so skiipping"
                     " to surface skeletonization")
                leftover_meshes_sig = [current_mesh]
    
            leftover_meshes_sig_surf_sk = []
            for m in tqdm(leftover_meshes_sig):
                surf_sk = generate_surface_skeleton(m.vertices,
                                               m.faces,
                                               surface_samples=n_surface_samples,
                                    n_surface_downsampling=n_surface_downsampling )
                if len(surf_sk) > 0:
                    leftover_meshes_sig_surf_sk.append(surf_sk)
            leftovers_stacked = stack_skeletons(leftover_meshes_sig_surf_sk)
            #print(f"significant_poisson_skeleton = {significant_poisson_skeleton}")
            #print(f"leftover_meshes_sig_surf_sk = {leftover_meshes_sig_surf_sk}")
            skeleton_ready_for_stitching = stack_skeletons([significant_poisson_skeleton,leftovers_stacked])
            
        #now want to stitch together whether generated from 
        if skeleton_print:
            print(f"After cgal process the un-stitched skeleton has shape {skeleton_ready_for_stitching.shape}")
        
        stitched_skeletons_full = stitch_skeleton(
                                                  skeleton_ready_for_stitching,
                                                  max_stitch_distance=max_stitch_distance,
                                                  stitch_print = False,
                                                  main_mesh = []
                                                )
        #stitched_skeletons_full_cleaned = clean_skeleton(stitched_skeletons_full)
        
        # erase the skeleton files if need to be
        if delete_temp_files:
            for sk_fi in skeleton_files:
                if Path(sk_fi).exists():
                    Path(sk_fi).unlink()
        
        # if created temp folder then erase if empty
        if str(output_folder.absolute()) == str(Path("./temp").absolute()):
            print("The process was using a temp folder")
            if len(list(output_folder.iterdir())) == 0:
                print("Temp folder was empty so deleting it")
                if output_folder.exists():
                    rmtree(str(output_folder.absolute()))
        
        return stitched_skeletons_full
    
def soma_skeleton_stitching(total_soma_skeletons,soma_mesh):
    """
    Purpose: Will stitch together the meshes that are touching
    the soma 
    
    Pseudocode: 
    1) Compute the soma mesh center point
    2) For meshes that were originally connected to soma
    a. Find the closest skeletal point to soma center
    b. Add an edge from closest point to soma center
    3) Then do stitching algorithm on all of remaining disconnected
        skeletons
    
    
    """
    # 1) Compute the soma mesh center point
    soma_center = np.mean(soma_mesh.vertices,axis=0)
    
    soma_connecting_skeleton = []
    for skel in total_soma_skeletons:
        #get the unique vertex points
        unique_skeleton_nodes = np.unique(skel.reshape(-1,3),axis=0)
        
        # a. Find the closest skeletal point to soma center
        # b. Add an edge from closest point to soma center
        mesh_tree = KDTree(unique_skeleton_nodes)
        distances,closest_node = mesh_tree.query(soma_center.reshape(-1,3))
        closest_skeleton_vert = unique_skeleton_nodes[closest_node[np.argmin(distances)]]
        soma_connecting_skeleton.append(np.array([closest_skeleton_vert,soma_center]).reshape(-1,2,3))
    
    print(f"soma_connecting_skeleton[0].shape = {soma_connecting_skeleton[0].shape}")
    print(f"total_soma_skeletons[0].shape = {total_soma_skeletons[0].shape}")
    # stith all of the ekeletons together
    soma_stitched_sk = stack_skeletons(total_soma_skeletons + soma_connecting_skeleton)
    
    return soma_stitched_sk



# ---- Util functions to be used for the skeletonization of soma containing meshes ------ #
def recursive_soma_skeletonization(main_mesh,
                                  soma_mesh_list,
                                soma_mesh_list_indexes,
                                   mesh_base_path="./temp_folder",
                                  soma_mesh_list_centers=[],
                                   current_name="segID"
                                  ):
    """
    Parameters:
    Mesh piece 
    The soma centers list and meshes list contained somewhere within the mesh piece
    
        Algorithm
    1) Start with the first soma and subtract from mesh
    2) Find all of the disconnected mesh pieces
    3) If there is still a soma piece that has not been processed, 
    find mesh pieces and all the somas that are contained within that
    4) Send Each one of those mesh pieces and soma lists
    to the recursive_soma_skeletonization (these will return skeletons)
    5) For all other pieces that do not have a soma do skeleton of branch
    6) Do soma skeleton stitching using all the branches and those returning
    from step 4
    7) return skeleton

    """
    print("\n\n Inside New skeletonization recursive calls\n\n")
    
    if len(soma_mesh_list) == 0:
        raise Exception("soma_mesh_list was empty")
    else:
        soma_mesh_list = list(soma_mesh_list)
    
    #0) If don't have the soma_mesh centers then calculate
    if len(soma_mesh_list_centers) != len(soma_mesh_list):
        soma_mesh_list_centers = find_soma_centroids(soma_mesh_list)
    
    #1) Start with the first soma and subtract from mesh
    #2) Find all of the disconnected mesh pieces
    current_soma = soma_mesh_list.pop(0)
    current_soma_index = soma_mesh_list_indexes.pop(0)
    current_soma_center = soma_mesh_list_centers.pop(0)
    mesh_pieces = subtract_soma(current_soma,main_mesh)
    print(f"currently working on soma index {current_soma_index}")
    
    print(f"mesh_pieces after the soma subtraction = {len(mesh_pieces)}")
    
    if len(mesh_pieces) < 1:
        #just return an empty list
        print("No significant pieces after soma cancellation so just using the soma center as the skeleton")
        return np.vstack([current_soma_center,current_soma_center]).reshape(-1,2,3)
        
    
    #3) If there is still a soma piece that has not been processed, 
    total_soma_skeletons = []
    
    if len(soma_mesh_list) > 0:
        #find mesh pieces and all the somas that are contained within that
        containing_mesh_indices = find_soma_centroid_containing_meshes(
                                            soma_mesh_list_centers,
                                            mesh_pieces
        )
        
        # rearrange into lists of somas per mesh soma 
        meshes_mapped_to_somas = grouping_containing_mesh_indices(containing_mesh_indices)
        
        #get all of the other mesh pieces that weren't a part of the soma containing
        mesh_pieces_with_soma = list(meshes_mapped_to_somas.keys())
        non_soma_branches = [k for i,k in enumerate(mesh_pieces) if i not in mesh_pieces_with_soma]

        print(f"meshes_mapped_to_somas = {meshes_mapped_to_somas}")
        
        #recursive call to the function to find all those skeletons for the 
        #mesh groupings 
        for mesh_idx,soma_list in meshes_mapped_to_somas.items():
            mesh_soma_list = [k for i,k in enumerate(soma_mesh_list) if i in soma_list]
            mesh_soma_list_indexes = [k for i,k in enumerate(soma_mesh_list_indexes) if i in soma_list]
            mesh_soma_list_centers = [k for i,k in enumerate(soma_mesh_list_centers) if i in soma_list]
            
            print(f"mesh_soma_list = {mesh_soma_list}\n"
                f"mesh_soma_list_indexes = {mesh_soma_list_indexes}\n"
                 f"mesh_soma_list_centers = {mesh_soma_list_centers}\n")
            
            soma_mesh_skeleton = recursive_soma_skeletonization(
                                  mesh_pieces[mesh_idx],
                                  soma_mesh_list=mesh_soma_list,
                                    soma_mesh_list_indexes = mesh_soma_list_indexes,
                                  soma_mesh_list_centers=mesh_soma_list_centers,
                                mesh_base_path=mesh_base_path,
                                current_name=current_name
            )
            
            total_soma_skeletons.append(soma_mesh_skeleton)
        
        
        
    
    else:
        non_soma_branches = mesh_pieces
    
    
    print(f"non_soma_branches = {len(non_soma_branches)}")
    print(f"mesh_pieces = {len(mesh_pieces)}")

    
    #5) For all other pieces that do not have a soma do skeleton of branch
    for dendrite_index,picked_dendrite in enumerate(non_soma_branches):
        dendrite_name=current_name + f"_soma_{current_soma_index}_branch_{dendrite_index}"
        
        print(f"\n\nWorking on {dendrite_name}")
        stitched_dendrite_skeleton = skeletonize_connected_branch(picked_dendrite,
                                                       output_folder=mesh_base_path,
                                                       name=dendrite_name,
                                                        skeleton_print = True)
        
        if len(stitched_dendrite_skeleton)<=0:
                print(f"*** Dendrite {dendrite_index} did not have skeleton computed***")
        else: 
            total_soma_skeletons.append(stitched_dendrite_skeleton)
    
    #stitching together the soma parts:
    soma_stitched_skeleton = soma_skeleton_stitching(total_soma_skeletons,current_soma)
    
    #return the stitched skeleton
    return soma_stitched_skeleton
    




    
def skeletonize_neuron(main_mesh_total,
                        segment_id = 12345,
                        soma_mesh_list = [],
                       mesh_base_path="",
                       current_name="",
                       filter_end_node_length=5000,
                       sig_th_initial_split=15,

                        ):
    """
    Purpose: to skeletonize a neuron
    
    Example of How to Use:
    
    neuron_file = '/notebooks/test_neurons/91216997676870145_excitatory_1.off'
    current_mesh = trimesh.load_mesh(neuron_file)
    segment_id = 91216997676870145
    html_path = neuron_file[:-4] + "_skeleton.html"
    current_mesh
    
    new_cleaned_skeleton = skeletonize_neuron(main_mesh_total=current_mesh,
                            segment_id = segment_id,
                           mesh_base_path="",
                           current_name="",

                            )

    new_cleaned_skeleton.shape
    
    """
    import skeleton_utils as sk
    global_time = time.time()
    
    #if no soma is provided then do own soma finding
    if len(soma_mesh_list) == 0:
        print("\nComputing Soma because none given")
        (soma_mesh_list, 
             run_time, 
             total_soma_list_sdf) = soma_utils.extract_soma_center(
                                segment_id,
                                main_mesh_total.vertices,
                                main_mesh_total.faces,
                                outer_decimation_ratio= 0.25,
                                large_mesh_threshold = 60000,
                                large_mesh_threshold_inner = 40000,
                                soma_width_threshold = 0.32,
                                soma_size_threshold = 20000,
                               inner_decimation_ratio = 0.25,
                               volume_mulitplier=7,
                               side_length_ratio_threshold=3,
                                soma_size_threshold_max=192000,
                                delete_files=True
            )
    else:
        print(f"Not computing soma because list already given: {soma_mesh_list}")
        
        
    
    if len(soma_mesh_list) <= 0:
        print(f"**** No Somas Found for Mesh {segment_id} so just one mesh")
        soma_mesh_list_centers = []
    else:
        #compute the soma centers
        print(f"Soma List = {soma_mesh_list}")
        
        soma_mesh_list_centers = find_soma_centroids(soma_mesh_list)
        print(f"soma_mesh_list_centers = {soma_mesh_list_centers}")

    
    split_meshes = split_significant_pieces(
                            main_mesh_total,
                            significance_threshold=sig_th_initial_split,
                            print_flag=False)
    
    
    """
    Pseudocode: 
    For all meshes in list
    1) compute soma center
    2) Find all the bounding boxes that contain the soma center
    3) Find the mesh with the closest distance from 
       one vertex to the soma center and tht is winner
    """
    
    
    #returns the index of the split_meshes index that contains each soma    
    containing_mesh_indices = find_soma_centroid_containing_meshes(soma_mesh_list_centers,
                                            split_meshes)
    
    non_soma_touching_meshes = [m for i,m in enumerate(split_meshes)
                     if i not in list(containing_mesh_indices.values())]
    
    
    #Adding the step that will filter away any pieces that are inside the soma
    if len(non_soma_touching_meshes) > 0 and len(soma_mesh_list) > 0:
        non_soma_touching_meshes = soma_utils.filter_away_inside_soma_pieces(soma_mesh_list,non_soma_touching_meshes,
                                        significance_threshold=sig_th_initial_split)
        
    
    print(f"# of non soma touching seperate meshes = {len(non_soma_touching_meshes)}")
    print(f"# of soma containing seperate meshes = {len(np.unique(list(containing_mesh_indices.values())))}")
    
    print(f"contents of soma containing seperate meshes = {np.unique(list(containing_mesh_indices.values()))}")
    
    
    # setting the base path and the current name
    if mesh_base_path == "":
        mesh_base_path = Path(f"./{segment_id}")
    else:
        mesh_base_path = Path(mesh_base_path)
        
    if current_name == "":
        current_name = f"{segment_id}"
        
    if mesh_base_path.exists():
        rmtree(str(mesh_base_path.absolute()))
    mesh_base_path.mkdir(parents=True,exist_ok=True)
    print(list(mesh_base_path.iterdir()))
    
    """
    Pseudocode for better skeletonization of the multi-soma cases:
    Have containing_mesh_indices that has the indices of the mesh that contain each soma
    
    Recursive function: 
    0) divide into soma indices that correspond to the same mesh indices
    For each group that corresponds to same mesh indices
    1) Start with the first soma and subtract from mesh
    2) Find all of the disconnected mesh pieces
    3) If there is still a soma piece that has not been processed, find the soma piece that is containing each soma and make into groups
    4) skeletonize all of the pieces that not have somas associated with them
    - if have lists from step 3, call the function for each of them, 
    4b) once recieve all of the skeletons then stitch together on that soma
    
    """
    
    
    
    #------ do the skeletonization of the soma touchings --------#
    print("\n\n ---- Working on soma touching skeletons ------")

    soma_touching_time = time.time()
    
    
    
    """ OLD WAY OF DOING THE SKELETONS FOR THE SOMA TOUCHING THAT DOES DOUBLE SKELETONIZATION 
    
    # ***** this part will have a repeat of the meshes that contain the soma *** #
    soma_touching_meshes = dict([(i,split_meshes[m_i]) 
                                 for i,m_i in containing_mesh_indices.items()])
    soma_touching_meshes_skeletons = []
    
    
    for s_i,main_mesh in soma_touching_meshes.items():
        #Do the mesh subtraction to get the disconnected pieces
        current_soma = soma_mesh_list[s_i]

        mesh_pieces = subtract_soma(current_soma,main_mesh)
        print(f"mesh_pieces after the soma subtraction = {len(mesh_pieces)}")
        #get each branch skeleton
        total_soma_skeletons = []
        for dendrite_index,picked_dendrite in enumerate(mesh_pieces):
            dendrite_name=current_name + f"_soma_{s_i}_branch_{dendrite_index}"
            print(f"\n\nWorking on {dendrite_name}")
            stitched_dendrite_skeleton = skeletonize_connected_branch(picked_dendrite,
                                                           output_folder=mesh_base_path,
                                                           name=dendrite_name,
                                                            skeleton_print = True)

            if len(stitched_dendrite_skeleton)<=0:
                print(f"*** Dendrite {dendrite_index} did not have skeleton computed***")
            else: 
                total_soma_skeletons.append(stitched_dendrite_skeleton)

    
    
    #stitching together the soma parts:
    soma_stitched_skeleton = soma_skeleton_stitching(total_soma_skeletons,current_soma)
    
    """
    
    
    # ---------------------- NEW WAY OF DOING THE SKELETONIZATION OF THE SOMA CONTAINING PIECES ------- #
    # rearrange into lists of somas per mesh soma 
    meshes_mapped_to_somas = grouping_containing_mesh_indices(containing_mesh_indices)

    print(f"meshes_mapped_to_somas = {meshes_mapped_to_somas}")

    soma_stitched_skeleton = []
    soma_mesh_list_indexes = list(np.arange(len(soma_mesh_list_centers)))
    
    #recursive call to the function to find all those skeletons for the 
    #mesh groupings 
    for mesh_idx,soma_list in meshes_mapped_to_somas.items():
        mesh_soma_list = [k for i,k in enumerate(soma_mesh_list) if i in soma_list]
        mesh_soma_list_indexes = [k for i,k in enumerate(soma_mesh_list_indexes) if i in soma_list]
        mesh_soma_list_centers = [k for i,k in enumerate(soma_mesh_list_centers) if i in soma_list]

        print(f"mesh_soma_list = {mesh_soma_list}\n"
            f"mesh_soma_list_indexes = {mesh_soma_list_indexes}\n"
             f"mesh_soma_list_centers = {mesh_soma_list_centers}\n")


        soma_mesh_skeleton = recursive_soma_skeletonization(
                                      split_meshes[mesh_idx],
                                      soma_mesh_list=mesh_soma_list,
                                        soma_mesh_list_indexes = mesh_soma_list_indexes,
                                      soma_mesh_list_centers=mesh_soma_list_centers,
                                    mesh_base_path=mesh_base_path,
                                    current_name=current_name
        )

        soma_stitched_skeleton.append(soma_mesh_skeleton)
        
    print(f"Total time for soma touching skeletons: {time.time() - soma_touching_time}")
    # ----------------------DONE WITH SKELETONIZATION OF THE SOMA CONTAINING PIECES ------- #
    
    
    #------ do the skeletonization of the NON soma touchings --------#
    print("\n\n ---- Working on non-soma touching skeletons ------")
    non_soma_time = time.time()

    non_soma_touching_meshes

    total_non_soma_skeletons = []
    for j,picked_non_soma_branch in enumerate(non_soma_touching_meshes):
    #     if j<66:
    #         continue
        dendrite_name=current_name + f"_non_soma_{j}"
        print(f"\n\nWorking on {dendrite_name}")
        stitched_dendrite_skeleton = skeletonize_connected_branch(picked_non_soma_branch,
                                                       output_folder=mesh_base_path,
                                                       name=dendrite_name,
                                                        skeleton_print = True)

        if len(stitched_dendrite_skeleton)<=0:
            print(f"*** Dendrite {dendrite_index} did not have skeleton computed***")
        else: 
            total_non_soma_skeletons.append(stitched_dendrite_skeleton)


    print(f"Time for non-soma skeletons = {time.time() - non_soma_time}")
    
    # --------- Doing the stitching of the skeletons -----------#
    try:
        stacked_non_soma_skeletons = stack_skeletons(total_non_soma_skeletons)
    except:
        print(f"stacked_non_soma_skeletons = {stacked_non_soma_skeletons}")
        raise Exception("stacked_non_soma_skeletons stack failed ")
    
    try:
        stacked_soma_skeletons = stack_skeletons(soma_stitched_skeleton)
    except:
        print(f"soma_stitched_skeleton = {soma_stitched_skeleton}")
        raise Exception("soma_stitched_skeleton stack failed ")
    
    
    try:
        whole_skeletons_for_stitching = stack_skeletons([stacked_non_soma_skeletons,stacked_soma_skeletons])
    except: 
        print(f"[stacked_non_soma_skeletons,stacked_soma_skeletons] = {[stacked_non_soma_skeletons,stacked_soma_skeletons]}")
        raise Exception("[stacked_non_soma_skeletons,stacked_soma_skeletons] stack failed")

    final_skeleton_pre_clean = stitch_skeleton(
                                                      whole_skeletons_for_stitching,
                                                      stitch_print = False,
                                                      main_mesh = []
                                                    )
    
    # --------  Doing the cleaning ------- #
    clean_time = time.time()
    new_cleaned_skeleton = clean_skeleton(final_skeleton_pre_clean,
                            distance_func=skeletal_distance,
                      min_distance_to_junction=filter_end_node_length,
                      return_skeleton=True,
                      print_flag=False)
    print(f"Total time for skeleton clean {time.time() - clean_time}")
    
    print(f"\n\n\n\nTotal time for whole skeletonization of neuron = {time.time() - global_time}")
    return new_cleaned_skeleton
