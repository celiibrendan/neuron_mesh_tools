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
sys.path.append("/meshAfterParty/meshAfterParty")

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


# In[ ]:


# minnie,schema = du.configure_minnie_vm()
# minnie.AutoProofreadNeurons4.drop()
# minnie.AutoProofreadStats4.drop()
# minnie.AutoProofreadSynapse4.drop()
# minnie.AutoProofreadSynapseErrors4.drop()
# minnie.schema.external['faces'].delete(delete_external_files=True)
# minnie.schema.external['skeleton'].delete(delete_external_files=True)


# # Proofreading Version

# In[ ]:


@schema
class AutoProofreadVersion(dj.Manual):
    definition="""
    proof_version      : tinyint unsigned                   # key by which to lookup the decomposition process version
    ---
    description          : varchar(256)                 # new parts of the iteration of the decomposition process
    """
versions=[[0,"exc,inh rules"],
         [1,"eliminated presyns on dendrite"],
         [2,"high fidelity axons"],
         [3,"improved crossover"],
         [4,"better crossover and more axon rules"]]


# dict_to_write = [dict(proof_version=k,description=v) for k,v in versions]
# AutoProofreadVersion.insert(dict_to_write,skip_duplicates=True)
AutoProofreadVersion()


# # Defining the Synapse Table

# In[ ]:


@schema
class AutoProofreadSynapse4(dj.Manual):
    definition="""
    synapse_id           : bigint unsigned              # synapse index within the segmentation
    synapse_type: enum('presyn','postsyn')
    ver                  : decimal(6,2)                 # the version number of the materializaiton
    ---
    segment_id           : bigint unsigned              # segment_id of the cell. Equivalent to Allen 'pt_root_id
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    nucleus_id           : int unsigned                 # id of nucleus from the flat segmentation  Equivalent to Allen: 'id'. 
    skeletal_distance_to_soma=NULL : double #the length (in um) of skeleton distance from synapse to soma (-1 if on the soma)
    """

@schema
class AutoProofreadSynapseErrors4(dj.Manual):
    definition="""
    synapse_id           : bigint unsigned              # synapse index within the segmentation
    synapse_type: enum('presyn','postsyn')
    ver                  : decimal(6,2)                 # the version number of the materializaiton
    ---
    segment_id           : bigint unsigned              # segment_id of the cell. Equivalent to Allen 'pt_root_id
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    nucleus_id           : int unsigned                 # id of nucleus from the flat segmentation  Equivalent to Allen: 'id'. 
    skeletal_distance_to_soma=NULL : double #the length (in um) of skeleton distance from synapse to soma (-1 if on the soma)
    """


# # Defining the Proofreading Stats Table

# In[ ]:


"""
This table will include the following information:

1) Filtering Info
2) Synapse Stats for Individual Neuron
3) Synapse Stats for Segment


**** thing need to add:
1) Axon faces
2) Axon length/area
2) Neuron faces
3) n_presyn_error_syn_non_axon
"""


# In[ ]:


@schema
class AutoProofreadStats4(dj.Manual):
    definition="""
    -> minnie.Decomposition()
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    -> minnie.AutoProofreadVersion()   # the version of code used for this cell typing classification
    ---
    mesh_faces: <faces>                      # faces indices that were saved off as belonging to proofread neuron (external storage)
    axon_faces: <faces>                      # faces indices that were saved off as belonging to proofread neuron's axon (external storage)
    
    axon_skeleton: <skeleton>      # the skeleton of the axon of the final proofread neuorn
    dendrite_skeleton: <skeleton>  # the skeleton of the dendrite branches of the final proofread neuorn
    neuron_skeleton: <skeleton>    # the skeleton of the entire neuron
    
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
    
    
    
    # ---------- new filters for v4 Stats ------------
    high_degree_branching_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    high_degree_branching_error_length=NULL : double #the length (in um) of skeleton distance canceled out by filter
    
    axon_webbing_t_merges_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    axon_webbing_t_merges_error_length=NULL : double #the length (in um) of skeleton distance canceled out by filter
    
    thick_t_merge_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    thick_t_merge_error_length=NULL : double #the length (in um) of skeleton distance canceled out by filter
    
    double_back_and_width_change_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    double_back_and_width_change_error_length=NULL : double #the length (in um) of skeleton distance canceled out by filter
    
    axon_fork_divergence_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    axon_fork_divergence_error_length=NULL : double #the length (in um) of skeleton distance canceled out by filter
    
    
    
    # ------------ For local valid synapses to that split_index
    n_valid_syn_presyn_for_split: int unsigned
    n_valid_syn_postsyn_for_split : int unsigned
    n_presyn_error_syn_non_axon :int unsigned
    
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
    
    limb_branch_to_cancel: longblob # stores the limb information from 
    red_blue_suggestions: longblob
    """
    
    


# # Creating the Auto Proofread Neuron Table

# In[ ]:


import numpy as np
import time
import classification_utils as clu

axon_version = 4
proof_version = 4

verbose = True

@schema
class AutoProofreadNeurons4(dj.Computed):
    definition="""
    -> minnie.Decomposition()
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    -> minnie.AutoProofreadVersion()             # the version of code used for this cell typing classification
    -> minnie.DecompositonAxonVersion()           # the version of the axon processing
    ---
    multiplicity  : tinyint unsigned   # the total number of neurons that came from the parent segment id
    # -------- Important Excitatory Inhibitory Classfication ------- #
    cell_type_predicted: enum('excitatory','inhibitory','other','unknown') # morphology predicted by classifier
    spine_category: enum('no_spined','sparsely_spined','densely_spined')
    
    n_axons: tinyint unsigned             # Number of axon candidates identified
    n_apicals: tinyint unsigned             # Number of apicals identified
    
    axon_length: double  # length (in um) of the classified axon skeleton
    axon_area: double # # area (in um^2) of the classified axon
    
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
    
    # ---- Stores Neuron Mesh Faces (moved to AutoProofreadStats) --------
    
    
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
    n_boutons: bigint

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
                             

    
    #key_source = minnie.Decomposition() & minnie.NucleiSegmentsRun2()
    #key_source = (minnie.Decomposition() & minnie.NucleiSegmentsRun2() 
    #              & minnie.DecompositionAxon().proj()) #& dict(segment_id=864691136361533410)
    key_source = (minnie.Decomposition() & minnie.NucleiSegmentsRun4() 
                  & minnie.DecompositionAxon().proj() 
              #& (minnie.AutoProofreadNeurons3() & "spine_category = 'densely_spined'").proj()
             ) #& dict(segment_id = 864691135575137566) #864691135575137566
    

    def make(self,key):
        """
        Pseudocode:
        1) Pull Down All of the Neurons
        2) Get the nucleus centers and the original mesh

        """
        
        # 1) Pull Down All of the Neurons
        segment_id = key["segment_id"]
        
        whole_pass_time = time.time()
        

        curr_output = pru.proofreading_table_processing(key,
                                  proof_version=proof_version,
                                  axon_version  = axon_version,
                                  compute_synapse_to_soma_skeletal_distance=True,
                                  perform_axon_classification = False,
                                  high_fidelity_axon_on_excitatory = False,
                                 verbose=True,)    
        # ------ Writing the Data To the Tables ----- #
            
            
        AutoProofreadSynapse_keys = curr_output["AutoProofreadSynapse_keys"]
        AutoProofreadSynapseErrors_keys = curr_output["AutoProofreadSynapseErrors_keys"]
        AutoProofreadNeurons_keys = curr_output["AutoProofreadNeurons_keys"]
        filtering_info_list = curr_output["filtering_info_list"]
        synapse_stats_list = curr_output["synapse_stats_list"]
        total_error_synapse_ids_list = curr_output["total_error_synapse_ids_list"]
        neuron_mesh_list = curr_output["neuron_mesh_list"]
        axon_mesh_list = curr_output["axon_mesh_list"]
        neuron_split_idxs = curr_output["neuron_split_idxs"]
        
        axon_skeleton_list = curr_output["axon_skeleton_list"]
        dendrite_skeleton_list = curr_output["dendrite_skeleton_list"]
        neuron_skeleton_list = curr_output["neuron_skeleton_list"]
            
        
        # Once have inserted all the new neurons need to compute the stats
        if verbose:
            print("Computing the overall stats")
            
        overall_syn_error_rates = pru.calculate_error_rate(total_error_synapse_ids_list,
                        synapse_stats_list,
                        verbose=True)
        
        
        # Final Part: Create the stats table entries and insert
        
        proofread_stats_entries = []
        
        stats_to_make_sure_in_proofread_stats = [
            
         'axon_on_dendrite_merges_error_area',
         'axon_on_dendrite_merges_error_length',
         'low_branch_clusters_error_area',
         'low_branch_clusters_error_length',
         'dendrite_on_axon_merges_error_area',
         'dendrite_on_axon_merges_error_length',
         'double_back_and_width_change_error_area',
         'double_back_and_width_change_error_length',
         'crossovers_error_area',
         'crossovers_error_length',
         'high_degree_coordinates_error_area',
         'high_degree_coordinates_error_length',
        ]
        
        
        for sp_idx,split_index in enumerate(neuron_split_idxs):
            
            #write the AutoProofreadNeurons and AutoProofreadSynapse Tabel
            keys_to_write = AutoProofreadSynapse_keys[sp_idx]
            AutoProofreadSynapse4.insert(keys_to_write,skip_duplicates=True)
            
            keys_to_write_errors = AutoProofreadSynapseErrors_keys[sp_idx]
            AutoProofreadSynapseErrors4.insert(keys_to_write_errors,skip_duplicates=True)
            
            
            
            new_key = AutoProofreadNeurons_keys[sp_idx]
            self.insert1(new_key,skip_duplicates=True,allow_direct_insert=True)
            
            synapse_stats = synapse_stats_list[sp_idx]
            filtering_info = filtering_info_list[sp_idx]
            limb_branch_to_cancel = pru.extract_from_filter_info(filtering_info,
                            name_to_extract="limb_branch_dict_to_cancel")
                            
            
            red_blue_suggestions = pru.extract_from_filter_info(filtering_info,
                            name_to_extract = "red_blue_suggestions")
            
            curr_key = dict(key,
                           split_index = split_index,
                           proof_version = proof_version,
                           
                             mesh_faces = neuron_mesh_list[sp_idx],
                            axon_faces = axon_mesh_list[sp_idx],
                            
                            axon_skeleton = axon_skeleton_list[sp_idx],
                            dendrite_skeleton = dendrite_skeleton_list[sp_idx],
                            neuron_skeleton = neuron_skeleton_list[sp_idx],
                         

                            # ------------ For local valid synapses to that split_index
                            n_valid_syn_presyn_for_split=synapse_stats["n_valid_syn_presyn"],
                            n_valid_syn_postsyn_for_split=synapse_stats["n_valid_syn_postsyn"],
                            n_presyn_error_syn_non_axon=synapse_stats["n_errored_syn_presyn_non_axon"],
                            
                            limb_branch_to_cancel = limb_branch_to_cancel,
                            red_blue_suggestions = red_blue_suggestions,
                           
                           
                           )
            
            
            for s in stats_to_make_sure_in_proofread_stats:
                if s not in filtering_info.keys():
                    curr_key[s] = None
            
            filter_key = {k:np.round(v,2) for k,v in filtering_info.items() if "area" in k or "length" in k}
            curr_key.update(filter_key)
            curr_key.update(overall_syn_error_rates)
            
            proofread_stats_entries.append(curr_key)
            
        
        AutoProofreadStats4.insert(proofread_stats_entries,skip_duplicates=True)
            
#         for pse in proofread_stats_entries:
#             AutoProofreadStats4.insert1(pse,skip_duplicates=True)
            

        print(f"\n\n ***------ Total time for {key['segment_id']} = {time.time() - whole_pass_time} ------ ***")


# # Running the Populate

# In[ ]:


curr_table = (minnie.schema.jobs & "table_name='__auto_proofread_neurons4'")
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
du = reload(du)
import random

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 800))
print('Populate Started')
if not test_mode:
    AutoProofreadNeurons4.populate(reserve_jobs=True, suppress_errors=True, order="random")
else:
    AutoProofreadNeurons4.populate(reserve_jobs=True, suppress_errors=False, order="random")
print('Populate Done')

print(f"Total time for AutoProofreadNeuron4 populate = {time.time() - start_time}")

