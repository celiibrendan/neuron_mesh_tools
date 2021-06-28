#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To save the validation synapse
tables for different validations

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


# # Setting up the virtual module

# In[ ]:


import system_utils as su
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


# In[ ]:


import meshlab
meshlab.set_meshlab_port(current_port=None)


# # Defining the Table

# In[ ]:


"""
What table to inherit from:
Decomposition Axon

Things want to save off:
1) validation_df
2) validation_df_ext
3) neuron object

For presyn/postsyn
- TP... counts
- scores


"""


# In[ ]:


# minnie.AutoProofreadValidationScore6.drop()
# minnie.schema.external['decomposition'].delete(delete_external_files=True)


# In[ ]:


import numpy as np
import time
import classification_utils as clu

import proofreading_utils as pru
import axon_utils as au
import validation_utils as vu

axon_version = au.axon_version

verbose = True

@schema
class AutoProofreadValidationScore6(dj.Computed):
    definition="""
    -> minnie.AutoProofreadValidationSegmentMap4()
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    ---
    decomposition        : <decomposition> # saved neuron object with high fidelity axon
    axon_length=NULL: double # axon length of the filtered neuron
    validation_df: longblob
    validation_df_ext: longblob #
    pre_tp: int unsigned #
    pre_tn: int unsigned
    pre_fp: int unsigned
    pre_fn: int unsigned
    
    pre_precision=NULL: double
    pre_recall=NULL: double
    pre_f1=NULL: double
    
    
    
    post_tp: int unsigned
    post_tn: int unsigned
    post_fp: int unsigned
    post_fn: int unsigned
    
    post_precision=NULL: double
    post_recall=NULL: double
    post_f1=NULL: double
    
    run_time=NULL : double                   # the amount of time to run (seconds)
    
    """
                             
    
    #key_source = minnie.Decomposition() & minnie.NucleiSegmentsRun2() & "segment_id=864691136540183458"
    pre_source = (minnie.AutoProofreadValidationSegmentMap4() & 
    (dj.U("old_segment_id") & minnie.DecompositionCellType.proj(old_segment_id="segment_id")))

    key_source = (pre_source - 
                  du.current_validation_segment_id_exclude.proj(old_segment_id="segment_id")
                  #& dict(old_segment_id=864691135373402824)
                 )
    

    def make(self,key):
        whole_pass_time = time.time()
        
        # ----------- Doing the v4 Processing ------- #
        
        segment_id = key["segment_id"]
        if verbose:
            print(f"\n-- Working on neuron {segment_id}---")

        segment_map_dict = (minnie.AutoProofreadValidationSegmentMap4() & dict(segment_id=segment_id)).fetch1()

        #1) Find the coordinates of the nucleus for that new segment
        nucleus_id = segment_map_dict["nucleus_id"]
        nuc_center_coords = du.nuclei_id_to_nucleus_centers(nucleus_id)
        if verbose:
            print(f"nuc_center_coords = {nuc_center_coords}")

        #2) Make sure that same number of DecompositionCellType objects as in Decomposition
        old_segment_id = segment_map_dict["old_segment_id"]
        if verbose:
            print(f"old_segment_id = {old_segment_id}")

        search_key = dict(segment_id=old_segment_id)
        n_somas = len(minnie.BaylorSegmentCentroid() & search_key)
        n_decomp_axon = len(minnie.DecompositionCellType() & search_key)
        if verbose:
            print(f"# of somas = {n_somas} and # of DecompositionCellType = {n_decomp_axon}")


        if n_somas != n_decomp_axon:
            raise Exception(f"# of somas = {n_somas} NOT MATCH # of DecompositionCellType = {n_decomp_axon}")

        #3) Pick the neuron object that is closest and within a certain range of the nucleus
        neuron_objs,split_idxs = du.decomposition_with_spine_recalculation(old_segment_id,
                                                                          attempt_apply_spines_to_neuron = False)
        if n_somas > 1:
            """
            Finding the closest soma:
            1) For each neuron object get the mesh center of the soma object
            2) Find the distance of each from the nucleus center
            3) Find the arg min distance and make sure within threshold
            4) Mark the current neuron and the current split index
            """
            nuclei_distance_threshold = 15000

            soma_center_coords = [k["S0"].mesh_center for k in neuron_objs]
            soma_distances = [np.linalg.norm(k-nuc_center_coords) for k in soma_center_coords]
            min_dist_arg = np.argmin(soma_distances)
            min_dist = soma_distances[min_dist_arg]

            if verbose:
                print(f"soma_distances = {soma_distances}")
                print(f"min_dist_arg = {min_dist_arg}, with min distance = {min_dist}")

            if min_dist > nuclei_distance_threshold:
                raise Exception(f"min_dist ({min_dist}) larger than nuclei_distance_threshold ({nuclei_distance_threshold})")

            neuron_obj = neuron_objs[min_dist_arg]
            split_index = split_idxs[min_dist_arg]

            if verbose:
                print(f"Winning split_index = {split_index}")
        else:
            split_index = split_idxs[0]
            neuron_obj = neuron_objs[0]

        (filt_neuron,
                     return_synapse_df_revised,
                     return_synapse_df_errors,
                    return_validation_df_revised,
                    return_validation_df_extension) =  vu.filtered_neuron_score(neuron_obj = neuron_obj,   
                                        filter_list = pru.v6_exc_filters(),
                                        plot_limb_branch_filter_with_disconnect_effect = False,
                                        verbose = True,
                                        plot_score=False,
                                        nucleus_id = nucleus_id,
                                        return_synapse_df_errors=True,
                                        return_validation_df_extension = True,                                        
                                        split_index=split_index)
        
        print(f"\n\n ----- Done Filtering ----------")
        
        
        
        #------- saving off the filtered neuron
        
        save_time = time.time()
        file_name = f"{filt_neuron.segment_id}_{filt_neuron.description}_v6_val"
        ret_file_path = filt_neuron.save_compressed_neuron(output_folder=str(du.get_decomposition_path()),
                                        file_name=file_name,        
                                          return_file_path=True,
                                         export_mesh=False,
                                         suppress_output=True)

        ret_file_path_str = str(ret_file_path.absolute()) + ".pbz2"
        print(f"Save time = {time.time() - save_time}")
        
        
        # ---------- Getting the scores of the proofreading ----- #
        presyn_scores_dict = vu.scores_presyn(return_validation_df_revised)
        postsyn_scores_dict = vu.scores_postsyn(return_validation_df_revised)

        cat = vu.synapse_validation_df_to_category_counts(return_validation_df_revised,
                                            print_postsyn=True,
                                            print_presyn=False)
        
        
        run_time = np.round(time.time() - whole_pass_time,2)
        
        final_dict = dict(key,
                          split_index = split_index,
                          
                          decomposition=ret_file_path_str,
                          axon_length = filt_neuron.axon_length,
                          
                          validation_df = return_validation_df_revised.to_numpy(),
                          validation_df_ext=return_validation_df_extension.to_numpy(),
                          
                          pre_tp=cat["presyn"]["TP"],
                            pre_tn=cat["presyn"]["TN"],
                            pre_fp=cat["presyn"]["FP"],
                            pre_fn=cat["presyn"]["FN"],

                            pre_precision=presyn_scores_dict["precision"],
                            pre_recall=presyn_scores_dict["recall"],
                            pre_f1=presyn_scores_dict["f1"],



                            post_tp=cat["postsyn"]["TP"],
                            post_tn=cat["postsyn"]["TN"],
                            post_fp=cat["postsyn"]["FP"],
                            post_fn=cat["postsyn"]["FN"],

                            post_precision=postsyn_scores_dict["precision"],
                            post_recall=postsyn_scores_dict["recall"],
                            post_f1=postsyn_scores_dict["f1"],
                          
                          
                          run_time = run_time
                         )
        
        self.insert1(final_dict,skip_duplicates=True,allow_direct_insert=True)
    
        print(f"\n\n ***------ Total time for {key['segment_id']} = {run_time} ------ ***")


# In[ ]:


curr_table = (minnie.schema.jobs & "table_name='__auto_proofread_validation_score6'")
#curr_table#.delete()
#curr_table.delete()


# In[ ]:


import time
import random
pru = reload(pru)
nru = reload(nru)
import neuron
neuron = reload(neuron)
import datajoint_utils as du
du = reload(du)

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 800))
print('Populate Started')
if not test_mode:
    AutoProofreadValidationScore6.populate(reserve_jobs=True, suppress_errors=True, order="random")
else:
    AutoProofreadValidationScore6.populate(reserve_jobs=True, suppress_errors=False,order="random")
print('Populate Done')

print(f"Total time for AutoProofreadValidationScore6 populate = {time.time() - start_time}")


# In[ ]:




