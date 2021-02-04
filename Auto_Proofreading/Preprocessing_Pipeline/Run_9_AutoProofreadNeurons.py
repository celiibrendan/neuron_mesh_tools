#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To decompose the multi-somas for splitting
using the new decomposition method



"""


# In[ ]:


import numpy as np
import datajoint as dj
import trimesh
from tqdm.notebook import tqdm
from pathlib import Path

from os import sys
sys.path.append("/meshAfterParty/")

import datajoint_utils as du
from importlib import reload


# In[ ]:


#so that it will have the adapter defined
from datajoint_utils import *


# In[ ]:


test_mode = False


# # Debugging the contains method

# In[ ]:


import system_utils as su


# In[ ]:


import minfig
import time
import numpy as np
#want to add in a wait for the connection part
random_sleep_sec = np.random.randint(0, 200)
print(f"Sleeping {random_sleep_sec} sec before conneting")
if not test_mode:
    time.sleep(random_sleep_sec)
print("Done sleeping")

du.config_celii()
du.set_minnie65_config_segmentation(minfig)
du.print_minnie65_config_paths(minfig)

#configuring will include the adapters
minnie,schema = du.configure_minnie_vm()


# In[ ]:


from importlib import reload

import neuron_utils as nru

import neuron

import neuron_visualizations as nviz

import time

import datajoint_utils as du

import numpy as np

import proofreading_utils as pru

import preprocessing_vp2 as pre

# -- For the axon classification --

import neuron_searching as ns

import skeleton_utils as sk

import numpy_utils as nu

import networkx_utils as xu

import system_utils as su

import classification_utils as clu
import proofreading_utils as pru

import datajoint as dj

from pykdtree.kdtree import KDTree
import trimesh_utils as tu
import proofreading_utils as pru
import numpy as np


# # Defining the Table

# In[ ]:


import neuron_utils as nru
import neuron
import trimesh_utils as tu
import numpy as np


# In[ ]:


import meshlab
meshlab.set_meshlab_port(current_port=None)


# # Defining the Synapse Table

# In[ ]:


@schema
class SynapseProofread(dj.Manual):
    definition="""
    synapse_id           : bigint unsigned              # synapse index within the segmentation
    synapse_type: enum('presyn','postsyn')
    ---
    segment_id           : bigint unsigned              # segment_id of the cell. Equivalent to Allen 'pt_root_id
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    nucleus_id           : int unsigned                 # id of nucleus from the flat segmentation  Equivalent to Allen: 'id'. 
    """


# # Defining the Proofreading Stats Table

# In[ ]:


"""
This table will include the following information:

1) Filtering Info
2) Synapse Stats for Individual Neuron
3) Synapse Stats for Segment



"""


# In[ ]:


@schema
class ProofreadStats(dj.Manual):
    definition="""
    -> minnie.Decomposition()
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    proof_version    : tinyint unsigned             # the version of code used for this cell typing classification
    ---
    
    axon_on_dendrite_merges_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    axon_on_dendrite_merges_error_length=NULL : double #the length (in um) of skeleton distance canceled out by filter
    
    low_branch_clusters_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    low_branch_clusters_error_length=NULL : double #the length (in um) of skeleton distance canceled out by filter
    
    dendrite_on_axon_merges_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    dendrite_on_axon_merges_error_length =NULL: double #the length (in um) of skeleton distance canceled out by filter
    
    double_back_and_width_change_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    double_back_and_width_change_error_length =NULL: double #the length (in um) of skeleton distance canceled out by filter
    
    crossovers_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    crossovers_error_length=NULL : double #the length (in um) of skeleton distance canceled out by filter
    
    high_degree_coordinates_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    high_degree_coordinates_error_length=NULL : double #the length (in um) of skeleton distance canceled out by filter
    
    # ------------ For local valid synapses to that split_index
    n_valid_syn_presyn_for_split: int unsigned
    n_valid_syn_postsyn_for_split : int unsigned
    
    # ------------ For global stats belonging to the whole segment
    # For the whole segment
    n_presyn_error_syn: int unsigned
    n_postsyn_error_syn: int unsigned
    total_error_synapses: int unsigned
    
    total_presyns: int unsigned 
    total_postsyns: int unsigned 
    total_synapses:int unsigned
    
    perc_error_presyn=NULL: double
    perc_error_postsyn=NULL: double
    
    overall_percent_error=NULL: double
    """
    
    


# # Creating the Auto Proofread Neuron Table

# In[ ]:


# minnie.AutoProofreadNeurons.drop()
# minnie.ProofreadStats.drop()
# minnie.SynapseProofread.drop()
# minnie.schema.external['faces'].delete(delete_external_files=True)


# In[ ]:


import numpy as np
import time
import classification_utils as clu

proof_version = 0

verbose = True

@schema
class AutoProofreadNeurons(dj.Computed):
    definition="""
    -> minnie.Decomposition()
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    proof_version    : tinyint unsigned             # the version of code used for this cell typing classification
    ---
    multiplicity  : tinyint unsigned   # the total number of neurons that came from the parent segment id
    # -------- Important Excitatory Inhibitory Classfication ------- #
    cell_type_predicted: enum('excitatory','inhibitory','other','unknown') # morphology predicted by classifier
    spine_category: enum('no_spined','sparsely_spined','densely_spined')
    
    n_axons: tinyint unsigned             # Number of axon candidates identified
    n_apicals: tinyint unsigned             # Number of apicals identified
    
    
    # ----- Soma Information ----#
    nucleus_id           : int unsigned                 # id of nucleus from the flat segmentation  Equivalent to Allen: 'id'.
    nuclei_distance      : double                    # the distance to the closest nuclei (even if no matching nuclei found)
    n_nuclei_in_radius   : tinyint unsigned          # the number of nuclei within the search radius of 15000 belonging to that segment
    n_nuclei_in_bbox     : tinyint unsigned          # the number of nuclei within the bounding box of that soma
    
    soma_x            : int unsigned                 # x coordinate of nucleus centroid in EM voxels (x: 4nm, y: 4nm, z: 40nm)
    soma_y            : int unsigned                 # y coordinate of nucleus centroid in EM voxels (x: 4nm, y: 4nm, z: 40nm)
    soma_z            : int unsigned                 # z coordinate of nucleus centroid in EM voxels (x: 4nm, y: 4nm, z: 40nm)
    
    max_soma_n_faces     : int unsigned                 # The largest number of faces of the somas
    max_soma_volume      : int unsigned                 # The largest volume of the somas the (volume in billions (10*9 nm^3))
    
    # ---- Stores Neuron Mesh Faces --------
    mesh_faces: <faces>                      # faces indices that were saved off as belonging to proofread neuron (external storage)
    
    
    # ------------- The Regular Neuron Information ----------------- #
    n_vertices           : int unsigned                 # number of vertices
    n_faces              : int unsigned                 # number of faces
    n_not_processed_soma_containing_meshes : int unsigned  #the number of meshes with somas that were not processed
    n_error_limbs: int #the number of limbs that are touching multiple somas or 1 soma in multiple places
    n_same_soma_multi_touching_limbs: int # number of limbs that touch the same soma multiple times
    n_multi_soma_touching_limbs: int # number of limbs that touch multiple somas
    n_somas: int #number of soma meshes detected
    n_limbs: int
    n_branches: int
    max_limb_n_branches=NULL:int
    
    skeletal_length=NULL: double
    max_limb_skeletal_length=NULL:double
    median_branch_length=NULL:double #gives information on average skeletal length to next branch point
    
    
    width_median=NULL: double #median width from mesh center without spines removed
    width_no_spine_median=NULL: double #median width from mesh center with spines removed
    width_90_perc=NULL: double # 90th percentile for width without spines removed
    width_no_spine_90_perc=NULL: double  # 90th percentile for width with spines removed
    
    
    n_spines: bigint

    spine_density=NULL: double # n_spines/ skeletal_length
    spines_per_branch=NULL: double
    
    skeletal_length_eligible=NULL: double # the skeletal length for all branches searched for spines
    n_spine_eligible_branches=NULL: int # the number of branches that were checked for spines because passed width threshold
    
    spine_density_eligible=NULL:double # n_spines/skeletal_length_eligible
    spines_per_branch_eligible=NULL:double # n_spines/n_spine_eligible_branches
    
    total_spine_volume=NULL: double # the sum of all spine volume
    spine_volume_median=NULL: double # median of the spine volume for those spines with able to calculate volume
    spine_volume_density=NULL: double #total_spine_volume/skeletal_length
    spine_volume_density_eligible=NULL: double #total_spine_volume/skeletal_length_eligible
    spine_volume_per_branch_eligible=NULL: double #total_spine_volume/n_spine_eligible_branches
    
    
    
    
    # ------ Information Used For Excitatory Inhibitory Classification -------- 
    axon_angle_maximum=NULL:double #the anlge of an identified axon
    spine_density_classifier:double              # the number of spines divided by skeletal length for branches analyzed in classification
    n_branches_processed: int unsigned                 # the number branches used for the spine density analysis
    skeletal_length_processed: double                 # The total skeletal length of the viable branches used for the spine density analysis
    n_branches_in_search_radius: int unsigned                 # the number branches existing in the search radius used for spine density
    skeletal_length_in_search_radius : double         # The total skeletal length of the branches existing in the search radius used for spine density
    
    

    run_time=NULL : double                   # the amount of time to run (seconds)

    
    """
                             
        
    
        
        
    
    
    
    key_source = minnie.Decomposition() & minnie.NucleiSegmentsRun2()
    
    

    def make(self,key):
        """
        Pseudocode:
        1) Pull Down All of the Neurons
        2) Get the nucleus centers and the original mesh

        """
        
        # 1) Pull Down All of the Neurons
        segment_id = key["segment_id"]
        
        print(f"\n\n------- AutoProofreadNeuron {segment_id}  ----------")
        
        neuron_objs,neuron_split_idxs = du.decomposition_with_spine_recalculation(segment_id)
        
        if verbose:
            print(f"Number of Neurons found ={len(neuron_objs)}")
        
        
        # 2)  ----- Pre-work ------

        nucleus_ids,nucleus_centers = du.segment_to_nuclei(segment_id)

        if verbose:
            print(f"Number of Corresponding Nuclei = {len(nucleus_ids)}")
            print(f"nucleus_ids = {nucleus_ids}")
            print(f"nucleus_centers = {nucleus_centers}")



        original_mesh = du.fetch_segment_id_mesh(segment_id)
        original_mesh_kdtree = KDTree(original_mesh.triangles_center)
        
        
        
        # 3) ----- Iterate through all of the Neurons and Proofread --------
        
        # lists to help save stats until write to ProofreadStats Table
        filtering_info_list = []
        synapse_stats_list = []
        total_error_synapse_ids_list = []
        
        
        for split_index,neuron_obj in zip(neuron_split_idxs,neuron_objs):
            
            whole_pass_time = time.time()
    
            if verbose:
                print(f"\n-----Working on Neuron Split {split_index}-----")


            # Part A: Proofreading the Neuron
            if verbose:
                print(f"\n   --> Part A: Proofreading the Neuron ----")


        #     nviz.visualize_neuron(neuron_obj,
        #                       limb_branch_dict="all")

            output_dict= pru.proofread_neuron(neuron_obj,
                                plot_limb_branch_filter_with_disconnect_effect=False,
                                plot_final_filtered_neuron=False,
                                verbose=False)

            filtered_neuron = output_dict["filtered_neuron"]
            cell_type_info = output_dict["cell_type_info"]
            filtering_info = output_dict["filtering_info"]

            
            


            # Part B: Getting Soma Centers and Matching To Nuclei
            if verbose:
                print(f"\n\n    --> Part B: Getting Soma Centers and Matching To Nuclei ----")


            winning_nucleus_id, nucleus_info = nru.pair_segment_id_to_nuclei(neuron_obj,
                                     "S0",
                                      nucleus_ids,
                                      nucleus_centers,
                                     nuclei_distance_threshold = 15000,
                                      return_matching_info = True,
                                     verbose=True)

            if verbose:
                print(f"nucleus_info = {nucleus_info}")
                print(f"winning_nucleus_id = {winning_nucleus_id}")

            





            # Part C: Getting the Faces of the Original Mesh
            if verbose:
                print(f"\n\n    --> Part C: Getting the Faces of the Original Mesh ----")

            original_mesh_faces = tu.original_mesh_faces_map(original_mesh,
                                                        filtered_neuron.mesh,
                                                        exact_match=True,
                                                        original_mesh_kdtree=original_mesh_kdtree)
            
            original_mesh_faces_file = du.save_proofread_faces(original_mesh_faces,
                                                              segment_id=segment_id,
                                                              split_index=split_index)

            

        #     nviz.plot_objects(recovered_mesh)






            # Part D: Getting the Synapse Information
            if verbose:
                print(f"\n\n    --> Part D: Getting the Synapse Information ----")


            (keys_to_write,
             synapse_stats,
             total_error_synapse_ids) = pru.synapse_filtering(filtered_neuron,
                            split_index,
                            nucleus_id=winning_nucleus_id,
                            segment_id=None,
                            return_synapse_filter_info = True,
                            return_synapse_center_data = False,
                            return_error_synapse_ids = True,
                            mapping_threshold = 500,
                              plot_synapses=False,
                            verbose = True,
                            original_mesh_method = True,
                            original_mesh = original_mesh,
                            original_mesh_kdtree = original_mesh_kdtree,
                            valid_faces_on_original_mesh=original_mesh_faces, 
                                                          
                            )


            



            soma_x,soma_y,soma_z = nru.soma_centers(filtered_neuron,
                                               soma_name="S0",
                                               voxel_adjustment=True)

        
        
        
            
            #7) Creating the dictionary to insert into the AutoProofreadNeuron
            new_key = dict(key,
                           split_index = split_index,
                           proof_version = proof_version,
                           
                           multiplicity = len(neuron_objs),
                           
                           # -------- Important Excitatory Inhibitory Classfication ------- #
                        cell_type_predicted = cell_type_info["inh_exc_class"],
                        spine_category=cell_type_info["spine_category"],

                        n_axons=cell_type_info["n_axons"],
                        n_apicals=cell_type_info["n_axons"],
                           
                           
                        
    
                        # ----- Soma Information ----#
                        nucleus_id         = nucleus_info["nuclei_id"],
                        nuclei_distance      = np.round(nucleus_info["nuclei_distance"],2),
                        n_nuclei_in_radius   = nucleus_info["n_nuclei_in_radius"],
                        n_nuclei_in_bbox     = nucleus_info["n_nuclei_in_bbox"],

                        soma_x           = soma_x,
                        soma_y           =soma_y,
                        soma_z           =soma_z,

                        # ---------- Mesh Faces ------ #
                        mesh_faces = original_mesh_faces_file,

                           
                        # ------------- The Regular Neuron Information (will be computed in the stats dict) ----------------- #
                        
                        
                        
                           # ------ Information Used For Excitatory Inhibitory Classification -------- 
                        axon_angle_maximum=cell_type_info["axon_angle_maximum"],
                        spine_density_classifier=cell_type_info["neuron_spine_density"],
                        n_branches_processed=cell_type_info["n_branches_processed"],
                        skeletal_length_processed=cell_type_info["skeletal_length_processed"],
                        n_branches_in_search_radius=cell_type_info["n_branches_in_search_radius"],
                        skeletal_length_in_search_radius=cell_type_info["skeletal_length_in_search_radius"],

                           
                        
                           
                           run_time=np.round(time.time() - whole_pass_time,4)
                          )
            
            
            
            
            
            
            
            stats_dict = filtered_neuron.neuron_stats()
            new_key.update(stats_dict)

            
            # ------ Writing the Data To the Tables ----- #
            SynapseProofread.insert(keys_to_write,skip_duplicates=True)
            
            self.insert1(new_key,skip_duplicates=True,allow_direct_insert=True)
            
            
            
            #saving following information for later processing:
            filtering_info_list.append(filtering_info)
            synapse_stats_list.append(synapse_stats)
            total_error_synapse_ids_list.append(total_error_synapse_ids)
            
            
        
        # Once have inserted all the new neurons need to compute the stats
        if verbose:
            print("Computing the overall stats")
            
        overall_syn_error_rates = pru.calculate_error_rate(total_error_synapse_ids_list,
                        synapse_stats_list,
                        verbose=True)
        
        
        # Final Part: Create the stats table entries and insert
        
        proofread_stats_entries = []
        
        
        
        for sp_idx,split_index in enumerate(neuron_split_idxs):
            synapse_stats = synapse_stats_list[sp_idx]
            filtering_info = filtering_info_list[sp_idx]
            
            curr_key = dict(key,
                           split_index = split_index,
                           proof_version = proof_version,
                           

                            # ------------ For local valid synapses to that split_index
                            n_valid_syn_presyn_for_split=synapse_stats["n_valid_syn_presyn"],
                            n_valid_syn_postsyn_for_split=synapse_stats["n_valid_syn_postsyn"],

                           
                           
                           )
            
            filter_key = {k:np.round(v,2) for k,v in filtering_info.items() if "area" in k or "length" in k}
            curr_key.update(filter_key)
            curr_key.update(overall_syn_error_rates)
            
            proofread_stats_entries.append(curr_key)
            
        
        ProofreadStats.insert(proofread_stats_entries,skip_duplicates=True)

            
        

        print(f"\n\n ***------ Total time for {key['segment_id']} = {time.time() - whole_pass_time} ------ ***")
    


# # Running the Populate

# In[ ]:


curr_table = (minnie.schema.jobs & "table_name='__auto_proofread_neurons'")
(curr_table)#.delete()# & "status='error'")
#curr_table.delete()
#(curr_table & "error_message = 'ValueError: need at least one array to concatenate'").delete()


# In[ ]:


import time
pru = reload(pru)
nru = reload(nru)
import neuron_searching as ns
ns = reload(ns)
clu = reload(clu)
import random

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 800))
print('Populate Started')
if not test_mode:
    AutoProofreadNeurons.populate(reserve_jobs=True, suppress_errors=False, order="random")
else:
    AutoProofreadNeurons.populate(reserve_jobs=True, suppress_errors=False, order="random")
print('Populate Done')

print(f"Total time for AutoProofreadNeuron populate = {time.time() - start_time}")


# In[ ]:




