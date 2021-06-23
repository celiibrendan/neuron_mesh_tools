#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To Create the table that
will store the neuron objects that have finer
axon preprocessing

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


import meshlab
meshlab.set_meshlab_port(current_port=None)


# # Proofreading Version

# In[ ]:


@schema
class DecompositonAxonVersion(dj.Manual):
    definition="""
    axon_version      : tinyint unsigned  # key by which to lookup the finer axon processing method
    ---
    description          : varchar(256)    # new parts of the finer axon preprocessing
    """
versions=[[0,"axon with standard meshparty"],
          [2,"axon with finer resolution"],
         [4,"even more fine resoution, axon skeleton, boutons, webbing"],
         [5,"filtered away floating pieces near soma for stitching"],
         [6, "max stitch distance = 2000, face threshold = 50"]]

dict_to_write = [dict(axon_version=k,description=v) for k,v in versions]
DecompositonAxonVersion.insert(dict_to_write,skip_duplicates=True)

DecompositonAxonVersion()


# In[ ]:


# minnie,schema = du.configure_minnie_vm()
# minnie.DecompositionAxon.delete()
# minnie.DecompositionCellType.delete()
# minnie.schema.external['decomposition'].delete(delete_external_files=True)


# In[ ]:


import numpy as np
import time
import classification_utils as clu
import proofreading_utils as pru
import axon_utils as au
import synapse_utils as syu
import preprocessing_vp2 as pre
import cell_type_utils as ctu

axon_version = au.axon_version
ver = 88

verbose = True
validation = True
inh_exc_class_to_use_for_axon = "Baylor"

@schema
class DecompositionCellType(dj.Computed):
    definition="""
    -> minnie.Decomposition()
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    -> minnie.DecompositonAxonVersion()             # the version of code used for this cell typing classification
    ---

    # -- attributes for the cell type classification ---
    decomposition        : <decomposition> # saved neuron object with high fidelity axon


    # ----- Nucleus Information ----#
    nucleus_id           : int unsigned                 # id of nucleus from the flat segmentation  Equivalent to Allen: 'id'.
    nuclei_distance      : double                    # the distance to the closest nuclei (even if no matching nuclei found)
    n_nuclei_in_radius   : tinyint unsigned          # the number of nuclei within the search radius of 15000 belonging to that segment
    n_nuclei_in_bbox     : tinyint unsigned          # the number of nuclei within the bounding box of that soma


    # ------ Information Used For Excitatory Inhibitory Classification (Baylor Cell) -------- 
    cell_type_predicted: enum('excitatory','inhibitory','other','unknown') # morphology predicted by classifier
    spine_category: enum('no_spined','sparsely_spined','densely_spined')

    n_axons: tinyint unsigned             # Number of axon candidates identified
    n_apicals: tinyint unsigned             # Number of apicals identified

    axon_angle_maximum=NULL:double #the anlge of an identified axon
    spine_density_classifier:double              # the number of spines divided by skeletal length for branches analyzed in classification
    n_branches_processed: int unsigned                 # the number branches used for the spine density analysis
    skeletal_length_processed: double                 # The total skeletal length of the viable branches used for the spine density analysis
    n_branches_in_search_radius: int unsigned                 # the number branches existing in the search radius used for spine density
    skeletal_length_in_search_radius : double         # The total skeletal length of the branches existing in the search radius used for spine density


    #---- allen classification info -----
    allen_e_i=NULL: enum('excitatory','inhibitory','other','unknown')
    allen_e_i_n_nuc=NULL: tinyint unsigned  
    allen_cell_type=NULL:varchar(256)
    allen_cell_type_n_nuc=NULL:tinyint unsigned  
    allen_cell_type_e_i=NULL:enum('excitatory','inhibitory','other','unknown')


    # ----- for the dendrite on axon filtering away --------
    dendrite_on_axon_merges_error_area=NULL : double #the area (in um ^ 2) of the faces canceled out by filter
    dendrite_on_axon_merges_error_length =NULL: double #the length (in um) of skeleton distance canceled out by filter

    # ----- attributes from the axon_features --- 
    cell_type_for_axon : enum('excitatory','inhibitory','other','unknown')
    axon_volume=NULL: double #volume of the oriented bounding box of axon (divided by 10^14)

    axon_length=NULL: double  # length (in um) of the classified axon skeleton
    axon_branch_length_median=NULL: double  # length (in um) of the classified axon skeleton
    axon_branch_length_mean=NULL: double  # length (in um) of the classified axon skeleton

    # number of branches in the axon
    axon_n_branches=NULL: int unsigned  
    axon_n_short_branches=NULL:  int unsigned
    axon_n_long_branches=NULL:  int unsigned
    axon_n_medium_branches=NULL:  int unsigned

    #bounding box features
    axon_bbox_x_min=NULL: double 
    axon_bbox_y_min=NULL: double 
    axon_bbox_z_min=NULL: double 
    axon_bbox_x_max=NULL: double 
    axon_bbox_y_max=NULL: double 
    axon_bbox_z_max=NULL: double 

    axon_bbox_x_min_soma_relative=NULL: double 
    axon_bbox_y_min_soma_relative=NULL: double 
    axon_bbox_z_min_soma_relative=NULL: double 
    axon_bbox_x_max_soma_relative=NULL: double 
    axon_bbox_y_max_soma_relative=NULL: double 
    axon_bbox_z_max_soma_relative=NULL: double 

    run_time=NULL : double                   # the amount of time to run (seconds)
    """
                             
    
    #key_source = minnie.Decomposition() & minnie.NucleiSegmentsRun2() & "segment_id=864691136540183458"
    #key_source = minnie.Decomposition() & du.proofreading_segment_id_restriction() & "segment_id=864691134884753146"
    key_source = (minnie.Decomposition() & 
                  du.current_validation_segment_id_restriction
                  - du.current_validation_segment_id_exclude)
    

    def make(self,key):
        """
        Pseudocode:
        1) Pull Down all the Neuron Objects associated with a segment_id
        
        For each neuron:
        2) Run the full axon preprocessing
        3) Save off the neuron
        4) Save dict entry to list
        
        
        5) Write the new entry to the table

        """
        
        
        # 1) Pull Down All of the Neurons
        segment_id = key["segment_id"]
        
        if verbose:
            print(f"------- Working on Neuron {segment_id} -----")
        
        whole_pass_time = time.time()
        
        #1) Pull Down all the Neuron Objects associated with a segment_id
        neuron_objs,neuron_split_idxs = du.decomposition_with_spine_recalculation(segment_id,
                                                                            ignore_DecompositionAxon=True,
                                                                            ignore_DecompositionCellType = True)

        if verbose:
            print(f"Number of Neurons found ={len(neuron_objs)}")

        #For each neuron:
        dict_to_write = []
        
        
        # -------- getting the nuclei info to match
#         ver = 88
#         nucleus_ids,nucleus_centers = du.segment_to_nuclei(segment_id,
#                                                                nuclei_version=ver)
        segment_map_dict = (minnie.AutoProofreadValidationSegmentMap4() & dict(old_segment_id=segment_id)).fetch1()
        nucleus_id = segment_map_dict["nucleus_id"]
        nuc_center_coords = du.nuclei_id_to_nucleus_centers(nucleus_id)
        
        nucleus_ids = [nucleus_id]
        nucleus_centers = [nuc_center_coords]
        
        print(f"nucleus_ids = {nucleus_ids}")
        print(f"nucleus_centers = {nucleus_centers}")
        
        for split_index,neuron_obj in zip(neuron_split_idxs,neuron_objs):
            
            if verbose:
                print(f"--> Working on Split Index {split_index} -----")
                
            st = time.time()
            
            
            # ------------- Does all of the processing -------------------
            
            #1) ------ Getting the paired nuclei ------
            winning_nucleus_id, nucleus_info = nru.pair_neuron_obj_to_nuclei(neuron_obj,
                                     "S0",
                                      nucleus_ids,
                                      nucleus_centers,
                                     nuclei_distance_threshold = 15000,
                                      return_matching_info = True,
                                     verbose=True)
            
            # else:
            #     winning_nucleus_id = 12345
            #     nucleus_info = dict()
            #     nucleus_info["nucleus_id"] = winning_nucleus_id
            #     nucleus_info["nuclei_distance"] = 0
            #     nucleus_info["n_nuclei_in_radius"] = 1
            #     nucleus_info["n_nuclei_in_bbox"] = 1

            if verbose:
                print(f"nucleus_info = {nucleus_info}")
                print(f"winning_nucleus_id = {winning_nucleus_id}")
        

            #2) ------- Finding the Allen Cell Types -------
            allen_cell_type_info = ctu.allen_nuclei_classification_info_from_nucleus_id(winning_nucleus_id)
            if verbose:
                print(f"allen_cell_type_info = {allen_cell_type_info}")

            
            #4) -------- Running the cell classification and stats--------------
            
            if verbose:
                print(f"\n\n ------ Part C: Inhibitory Excitatory Classification ---- \n\n")

            filter_time = time.time()

            (inh_exc_class,
             spine_category,
             axon_angles,
             n_axons,
             n_apicals,
             neuron_spine_density,
             n_branches_processed,
             skeletal_length_processed,
             n_branches_in_search_radius,
             skeletal_length_in_search_radius
             ) = clu.inhibitory_excitatory_classifier(neuron_obj,
                                                return_spine_classification=True,
                                                return_axon_angles=True,
                                                 return_n_axons=True,
                                                 return_n_apicals=True,
                                                 return_spine_statistics=True,
                                                     axon_limb_branch_dict_precomputed=None,
                                                axon_angles_precomputed=None,
                                                     verbose=verbose)
            if verbose:
                print(f"Total time for classification = {time.time() - filter_time}")

            all_axon_angles = []
            for limb_idx,limb_data in axon_angles.items():
                for candidate_idx,cand_angle in limb_data.items():
                    all_axon_angles.append(cand_angle)

            if len(axon_angles)>0:
                axon_angle_maximum = np.max(all_axon_angles)
            else:
                axon_angle_maximum = 0


            if verbose:
                print("\n -- Cell Type Classification Results --")
                print(f"inh_exc_class={inh_exc_class}")
                print(f"spine_category={spine_category}")
                print(f"axon_angles={axon_angles}")
                print(f"n_axons={n_axons}")
                print(f"n_apicals={n_apicals}")
                print(f"neuron_spine_density={neuron_spine_density}")
                print(f"n_branches_processed={n_branches_processed}")
                print(f"skeletal_length_processed={skeletal_length_processed}")
                print(f"n_branches_in_search_radius={n_branches_in_search_radius}")
                print(f"skeletal_length_in_search_radius={skeletal_length_in_search_radius}")

            baylor_cell_type_info = dict(
                        cell_type_predicted=inh_exc_class,
                         spine_category=spine_category,
                        axon_angle_maximum = axon_angle_maximum,
                         n_axons=n_axons,
                         n_apicals=n_apicals,
                         spine_density_classifier=neuron_spine_density,
                         n_branches_processed=neuron_spine_density,
                         skeletal_length_processed=skeletal_length_processed,
                         n_branches_in_search_radius=n_branches_in_search_radius,
                         skeletal_length_in_search_radius=skeletal_length_in_search_radius,

            )

            
            
            #5) ----- Deciding on cell type to use for axon 
            e_i_class = inh_exc_class
            if inh_exc_class_to_use_for_axon == "Allen" and allen_cell_type_info["e_i"] is not None:
                e_i_class = allen_cell_type_info["e_i"]

            if verbose:
                print(f"e_i_class = {e_i_class} with inh_exc_class_to_use_for_axon = {inh_exc_class_to_use_for_axon}")

            
            
            #6) -------- If excitatory running the axon processing--------------
            """
            Psuedocode: 
            If e_i class is excitatory:
            1) Filter away the axon on dendrite
            2) Do the higher fidelity axon processing
            3) Compute the axon features

            """
            
            if e_i_class == "excitatory" and neuron_obj.axon_limb_name is not None:
                if verbose:
                    print(f"Excitatory so performing high fidelity axon and computing axon features")
            #     1) Filter away the axon on dendrite
            #     2) Do the higher fidelity axon processing

                o_neuron, filtering_info = au.complete_axon_processing(neuron_obj,
                                                                      perform_axon_classification = False,
                                                                      return_filtering_info = True)
                filtering_info = {k:np.round(v,2) for k,v in filtering_info.items() if "area" in k or "length" in k}
                #3) Compute the axon features
                axon_features = au.axon_features_from_neuron_obj(o_neuron)
            else:
                nru.clear_all_branch_labels(neuron_obj,labels_to_clear="axon")
                o_neuron = neuron_obj
                axon_features = dict()
                filtering_info = dict()


            
            #3) ------ Adding the Synapses -----------
            o_neuron = syu.add_synapses_to_neuron_obj(o_neuron,
                            validation = validation,
                            verbose  = True,
                            original_mesh = None,
                            plot_valid_error_synapses = False,
                            calculate_synapse_soma_distance = False,
                            add_valid_synapses = True,
                              add_error_synapses=False)
            
            
            
            # ------- Saving off the neuron object ----------------
            save_time = time.time()
            ret_file_path = o_neuron.save_compressed_neuron(
                                            output_folder=str(du.get_decomposition_path()),
                                            #output_folder = "./",
            file_name=f"{o_neuron.segment_id}_{split_index}_split_axon_v{au.axon_version}",
                                              return_file_path=True,
                                             export_mesh=False,
                                             suppress_output=True)

            ret_file_path_str = str(ret_file_path.absolute()) + ".pbz2"
            
            if verbose:
                print(f"ret_file_path_str = {ret_file_path_str}")
                print(f"Save time = {time.time() - save_time}")
            
            n_dict = dict(key,
              split_index = split_index,
              axon_version = au.axon_version,
             decomposition=ret_file_path_str,
              run_time = np.round(time.time() - st,2),
            cell_type_for_axon = e_i_class,
             )
            
            dicts_for_update = [baylor_cell_type_info,
                                allen_cell_type_info,
                                nucleus_info,
                                filtering_info,
                                axon_features]
            
            for d in dicts_for_update:
                n_dict.update(d)

            self.insert1(n_dict,skip_duplicates=True,allow_direct_insert=True)
            #dict_to_write.append(n_dict)
        
        #write the
        #self.insert(dict_to_write,skip_duplicates=True,allow_direct_insert=True)

        print(f"\n\n ***------ Total time for {key['segment_id']} = {time.time() - whole_pass_time} ------ ***")


# # Running the Populate

# In[ ]:


curr_table = (minnie.schema.jobs & "table_name='__decomposition_cell_type'")
curr_table#.delete()
#(curr_table & "status='error'") #& "error_message='IndexError: list index out of range'"


# In[ ]:


import time
pru = reload(pru)
nru = reload(nru)
import neuron_searching as ns
ns = reload(ns)
clu = reload(clu)
du = reload(du)
ctu = reload(ctu)
import random

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 800))
print('Populate Started')
if not test_mode:
    DecompositionCellType.populate(reserve_jobs=True, suppress_errors=True, order="random")
else:
    DecompositionCellType.populate(reserve_jobs=True, suppress_errors=False,)# order="random")
print('Populate Done')

print(f"Total time for DecompositionCellType populate = {time.time() - start_time}")


# In[ ]:




