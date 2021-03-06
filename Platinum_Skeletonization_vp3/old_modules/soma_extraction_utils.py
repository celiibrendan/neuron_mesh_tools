import cgal_Segmentation_Module as csm
from whole_neuron_classifier_datajoint_adapted import extract_branches_whole_neuron
import whole_neuron_classifier_datajoint_adapted as wcda 
import time
import trimesh
import numpy as np
import datajoint as dj
import os
from meshlab import Decimator , Poisson
from pathlib import Path
from pykdtree.kdtree import KDTree


"""
Checking the new validation checks
"""
def side_length_ratios(current_mesh):
    """
    Will compute the ratios of the bounding box sides
    To be later used to see if there is skewness
    """

    # bbox = current_mesh.bounding_box_oriented.vertices
    bbox = current_mesh.bounding_box_oriented.vertices
    x_axis_unique = np.unique(bbox[:,0])
    y_axis_unique = np.unique(bbox[:,1])
    z_axis_unique = np.unique(bbox[:,2])
    x_length = (np.max(x_axis_unique) - np.min(x_axis_unique)).astype("float")
    y_length = (np.max(y_axis_unique) - np.min(y_axis_unique)).astype("float")
    z_length = (np.max(z_axis_unique) - np.min(z_axis_unique)).astype("float")
    #print(x_length,y_length,z_length)
    #compute the ratios:
    xy_ratio = float(x_length/y_length)
    xz_ratio = float(x_length/z_length)
    yz_ratio = float(y_length/z_length)
    side_ratios = [xy_ratio,xz_ratio,yz_ratio]
    flipped_side_ratios = []
    for z in side_ratios:
        if z < 1:
            flipped_side_ratios.append(1/z)
        else:
            flipped_side_ratios.append(z)
    return flipped_side_ratios

def side_length_check(current_mesh,side_length_ratio_threshold=3):
    side_length_ratio_names = ["xy","xz","yz"]
    side_ratios = side_length_ratios(current_mesh)
    pass_threshold = [(k <= side_length_ratio_threshold) and
                      (k >= 1/side_length_ratio_threshold) for k in side_ratios]
    for i,(rt,truth) in enumerate(zip(side_ratios,pass_threshold)):
        if not truth:
            print(f"{side_length_ratio_names[i]} = {rt} ratio was beyong {side_length_ratio_threshold} multiplier")

    if False in pass_threshold:
        return False
    else:
        return True

import random

def largest_mesh_piece(msh):
    mesh_splits_inner = msh.split(only_watertight=False)
    total_mesh_split_lengths_inner = [len(k.faces) for k in mesh_splits_inner]
    ordered_mesh_splits_inner = mesh_splits_inner[np.flip(np.argsort(total_mesh_split_lengths_inner))]
    return ordered_mesh_splits_inner[0]
def soma_volume_ratio(current_mesh):
    """
    bounding_box_oriented: rotates the box to be less volume
    bounding_box : does not rotate the box and makes it axis aligned
    
    ** checks to see if closed mesh and if not then make closed **
    """
    poisson_temp_folder = Path.cwd() / "Poisson_temp"
    poisson_temp_folder.mkdir(parents=True,exist_ok=True)
    with Poisson(poisson_temp_folder,overwrite=True) as Poisson_obj_temp:

        #get the largest piece
        lrg_mesh = largest_mesh_piece(current_mesh)
        if not lrg_mesh.is_watertight:
            print("Using Poisson Surface Reconstruction to make mesh watertight")
            #run the Poisson Surface reconstruction and get the largest piece
            new_mesh_inner,poisson_file_obj = Poisson_obj_temp(vertices=lrg_mesh.vertices,
                   faces=lrg_mesh.faces,
                   return_mesh=True,
                   delete_temp_files=True,
                   segment_id=random.randint(0,999999))
            lrg_mesh = largest_mesh_piece(new_mesh_inner)

        #turn the mesh into a closed mesh based on 

        ratio_val = lrg_mesh.bounding_box.volume/lrg_mesh.volume
    #     if ratio_val < 1:
    #         raise Exception("Less than 1 value in volume ratio computation")
    
    return ratio_val

def soma_volume_check(current_mesh,multiplier=8):
    ratio_val= soma_volume_ratio(current_mesh)
    print("Inside sphere validater: ratio_val = " + str(ratio_val))
    if np.abs(ratio_val) > multiplier:
        return False
    return True



# -------------- Function that will extract the soma ------- #
def extract_soma_center(segment_id,
                        current_mesh_verts,
                        current_mesh_faces,
                       outer_decimation_ratio= 0.25,
                        large_mesh_threshold = 60000,
                        large_mesh_threshold_inner = 40000,
                        soma_width_threshold = 0.32,
                        soma_size_threshold = 20000,
                       inner_decimation_ratio = 0.25,
                       volume_mulitplier=8,
                       side_length_ratio_threshold=3,
                       soma_size_threshold_max=192000, #this puts at 12000 once decimated
                       delete_files=True
                       ):    
    """
    Will extract the soma meshes (possible multiple) from
    a single mesh

    """
    
    global_start_time = time.time()

    #Adjusting the thresholds based on the decimations
    large_mesh_threshold = large_mesh_threshold*outer_decimation_ratio
    large_mesh_threshold_inner = large_mesh_threshold_inner*outer_decimation_ratio
    soma_size_threshold = soma_size_threshold*outer_decimation_ratio
    soma_size_threshold_max = soma_size_threshold_max*outer_decimation_ratio

    #adjusting for inner decimation
    soma_size_threshold = soma_size_threshold*inner_decimation_ratio
    soma_size_threshold_max = soma_size_threshold_max*inner_decimation_ratio
    print(f"Current Arguments Using (adjusted for decimation):\n large_mesh_threshold= {large_mesh_threshold}"
                 f" \nlarge_mesh_threshold_inner = {large_mesh_threshold_inner}"
                  f" \nsoma_size_threshold = {soma_size_threshold}"
                 f" \nsoma_size_threshold_max = {soma_size_threshold_max}"
                 f"\nouter_decimation_ratio = {outer_decimation_ratio}"
                 f"\ninner_decimation_ratio = {inner_decimation_ratio}")


    # ------------------------------


    temp_folder = f"./{segment_id}"
    temp_object = Path(temp_folder)
    #make the temp folder if it doesn't exist
    temp_object.mkdir(parents=True,exist_ok=True)

    #making the decimation and poisson objections
    Dec_outer = Decimator(outer_decimation_ratio,temp_folder,overwrite=True)
    Dec_inner = Decimator(inner_decimation_ratio,temp_folder,overwrite=True)
    Poisson_obj = Poisson(temp_folder,overwrite=True)

    #Step 1: Decimate the Mesh and then split into the seperate pieces
    new_mesh,output_obj = Dec_outer(vertices=current_mesh_verts,
             faces=current_mesh_faces,
             segment_id=segment_id,
             return_mesh=True,
             delete_temp_files=False)

    #preforming the splits of the decimated mesh

    mesh_splits = new_mesh.split(only_watertight=False)

    #get the largest mesh
    mesh_lengths = np.array([len(split.faces) for split in mesh_splits])


    total_mesh_split_lengths = [len(k.faces) for k in mesh_splits]
    ordered_mesh_splits = mesh_splits[np.flip(np.argsort(total_mesh_split_lengths))]
    list_of_largest_mesh = [k for k in ordered_mesh_splits if len(k.faces) > large_mesh_threshold]

    print(f"Total found significant pieces before Poisson = {list_of_largest_mesh}")

    #if no significant pieces were found then will use smaller threshold
    if len(list_of_largest_mesh)<=0:
        print(f"Using smaller large_mesh_threshold because no significant pieces found with {large_mesh_threshold}")
        list_of_largest_mesh = [k for k in ordered_mesh_splits if len(k.faces) > large_mesh_threshold/2]

    total_soma_list = []
    total_classifier_list = []
    total_poisson_list = []
    total_soma_list_sdf = []

    #start iterating through where go through all pieces before the poisson reconstruction
    no_somas_found_in_big_loop = 0
    for i,largest_mesh in enumerate(list_of_largest_mesh):
        print(f"----- working on large mesh #{i}: {largest_mesh}")

        somas_found_in_big_loop = False

        largest_file_name = str(output_obj.stem) + "_largest_piece.off"
        pre_largest_mesh_path = temp_object / Path(str(output_obj.stem) + "_largest_piece.off")
        pre_largest_mesh_path = pre_largest_mesh_path.absolute()
        print(f"pre_largest_mesh_path = {pre_largest_mesh_path}")
        # ******* This ERRORED AND CALLED OUR NERUON NONE: 77697401493989254 *********
        new_mesh_inner,poisson_file_obj = Poisson_obj(vertices=largest_mesh.vertices,
                   faces=largest_mesh.faces,
                   return_mesh=True,
                   mesh_filename=largest_file_name,
                   delete_temp_files=False)


        #splitting the Poisson into the largest pieces and ordering them
        mesh_splits_inner = new_mesh_inner.split(only_watertight=False)
        total_mesh_split_lengths_inner = [len(k.faces) for k in mesh_splits_inner]
        ordered_mesh_splits_inner = mesh_splits_inner[np.flip(np.argsort(total_mesh_split_lengths_inner))]

        list_of_largest_mesh_inner = [k for k in ordered_mesh_splits_inner if len(k.faces) > large_mesh_threshold_inner]
        print(f"Total found significant pieces AFTER Poisson = {list_of_largest_mesh_inner}")

        n_failed_inner_soma_loops = 0
        for j, largest_mesh_inner in enumerate(list_of_largest_mesh_inner):
            print(f"----- working on mesh after poisson #{j}: {largest_mesh_inner}")

            largest_mesh_path_inner = str(poisson_file_obj.stem) + "_largest_inner.off"

            #Decimate the inner poisson piece
            largest_mesh_path_inner_decimated,output_obj_inner = Dec_inner(
                                vertices=largest_mesh_inner.vertices,
                                 faces=largest_mesh_inner.faces,
                                mesh_filename=largest_mesh_path_inner,
                                 return_mesh=True,
                                 delete_temp_files=False)

            print(f"done exporting decimated mesh: {largest_mesh_path_inner}")

            faces = np.array(largest_mesh_path_inner_decimated.faces)
            verts = np.array(largest_mesh_path_inner_decimated.vertices)

            segment_id_new = int(str(segment_id) + f"{i}{j}")

            verts_labels, faces_labels, soma_value,classifier = wcda.extract_branches_whole_neuron(
                                    import_Off_Flag=False,
                                    segment_id=segment_id_new,
                                    vertices=verts,
                                     triangles=faces,
                                    pymeshfix_Flag=False,
                                     import_CGAL_Flag=False,
                                     return_Only_Labels=True,
                                     clusters=3,
                                     smoothness=0.2,
                                    soma_only=True,
                                    return_classifier = True
                                    )
            print(f"soma_sdf_value = {soma_value}")

            total_classifier_list.append(classifier)
            #total_poisson_list.append(largest_mesh_path_inner_decimated)

            # Save all of the portions that resemble a soma
            median_values = np.array([v["median"] for k,v in classifier.sdf_final_dict.items()])
            segmentation = np.array([k for k,v in classifier.sdf_final_dict.items()])

            #order the compartments by greatest to smallest
            sorted_medians = np.flip(np.argsort(median_values))
            print(f"segmentation[sorted_medians],median_values[sorted_medians] = {(segmentation[sorted_medians],median_values[sorted_medians])}")
            print(f"Sizes = {[classifier.sdf_final_dict[g]['n_faces'] for g in segmentation[sorted_medians]]}")

            valid_soma_segments_width = [g for g,h in zip(segmentation[sorted_medians],median_values[sorted_medians]) if ((h > soma_width_threshold)
                                                                and (classifier.sdf_final_dict[g]["n_faces"] > soma_size_threshold)
                                                                and (classifier.sdf_final_dict[g]["n_faces"] < soma_size_threshold_max))]
            valid_soma_segments_sdf = [h for g,h in zip(segmentation[sorted_medians],median_values[sorted_medians]) if ((h > soma_width_threshold)
                                                                and (classifier.sdf_final_dict[g]["n_faces"] > soma_size_threshold)
                                                                and (classifier.sdf_final_dict[g]["n_faces"] < soma_size_threshold_max))]

            print("valid_soma_segments_width")
            to_add_list = []
            to_add_list_sdf = []
            if len(valid_soma_segments_width) > 0:
                print(f"      ------ Found {len(valid_soma_segments_width)} viable somas: {valid_soma_segments_width}")
                somas_found_in_big_loop = True
                #get the meshes only if signfiicant length
                labels_list = classifier.labels_list

                for v,sdf in zip(valid_soma_segments_width,valid_soma_segments_sdf):
                    submesh_face_list = np.where(classifier.labels_list == v)[0]
                    soma_mesh = largest_mesh_path_inner_decimated.submesh([submesh_face_list],append=True)

                    
                    if side_length_check(soma_mesh,side_length_ratio_threshold) and soma_volume_check(soma_mesh,volume_mulitplier):
                        to_add_list.append(soma_mesh)
                        to_add_list_sdf.append(sdf)
                        
                    else:
                        print(f"--->This soma mesh was not added because it did not pass the sphere validation: {soma_mesh}")

                n_failed_inner_soma_loops = 0

            else:
                n_failed_inner_soma_loops += 1

            total_soma_list_sdf += to_add_list_sdf
            total_soma_list += to_add_list

            # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
            if n_failed_inner_soma_loops >= 2:
                print("breaking inner loop because 2 soma fails in a row")
                break


        # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
        if somas_found_in_big_loop == False:
            no_somas_found_in_big_loop += 1
            if no_somas_found_in_big_loop >= 2:
                print("breaking because 2 fails in a row in big loop")
                break

        else:
            no_somas_found_in_big_loop = 0





    """ IF THERE ARE MULTIPLE SOMAS THAT ARE WITHIN A CERTAIN DISTANCE OF EACH OTHER THEN JUST COMBINE THEM INTO ONE"""
    pairings = []
    for y,soma_1 in enumerate(total_soma_list):
        for z,soma_2 in enumerate(total_soma_list):
            if y<z:
                mesh_tree = KDTree(soma_1.vertices)
                distances,closest_node = mesh_tree.query(soma_2.vertices)

                if np.min(distances) < 4000:
                    pairings.append([y,z])


    #creating the combined meshes from the list
    total_soma_list_revised = []
    total_soma_list_revised_sdf = []
    if len(pairings) > 0:
        """
        Pseudocode: 
        Use a network function to find components

        """


        import networkx as nx
        new_graph = nx.Graph()
        new_graph.add_edges_from(pairings)
        grouped_somas = list(nx.connected_components(new_graph))

        somas_being_combined = []
        print(f"There were soma pairings: Connected components in = {grouped_somas} ")
        for comp in grouped_somas:
            comp = list(comp)
            somas_being_combined += list(comp)
            current_mesh = total_soma_list[comp[0]]
            for i in range(1,len(comp)):
                current_mesh += total_soma_list[comp[i]]

            total_soma_list_revised.append(current_mesh)
            #where can average all of the sdf values
            total_soma_list_revised_sdf.append(np.min(np.array(total_soma_list_sdf)[comp]))

        #add those that weren't combined to total_soma_list_revised
        leftover_somas = [total_soma_list[k] for k in range(0,len(total_soma_list)) if k not in somas_being_combined]
        leftover_somas_sdfs = [total_soma_list_sdf[k] for k in range(0,len(total_soma_list)) if k not in somas_being_combined]
        if len(leftover_somas) > 0:
            total_soma_list_revised += leftover_somas
            total_soma_list_revised_sdf += leftover_somas_sdfs
            
        print(f"Final total_soma_list_revised = {total_soma_list_revised}")
        print(f"Final total_soma_list_revised_sdf = {total_soma_list_revised_sdf}")


    if len(total_soma_list_revised) == 0:
        total_soma_list_revised = total_soma_list
        total_soma_list_revised_sdf = total_soma_list_sdf

    run_time = time.time() - global_start_time

    print(f"\n\n\n Total time for run = {time.time() - global_start_time}")

    #need to erase all of the temporary files ******
    #import shutil
    #shutil.rmtree(directory)

    """
    Need to delete all files in the temp folder *****
    """
    
    if delete_files:
        #now erase all of the files used
        from shutil import rmtree

        #remove the directory with the meshes
        rmtree(str(temp_object.absolute()))

        #removing the temporary files
        temp_folder = Path("./temp")
        temp_files = [x for x in temp_folder.glob('**/*')]
        seg_temp_files = [x for x in temp_files if str(segment_id) in str(x)]

        for f in seg_temp_files:
            f.unlink()

    #return total_soma_list, run_time
    return total_soma_list_revised,run_time,total_soma_list_revised_sdf

