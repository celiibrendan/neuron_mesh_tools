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
import axon_utils as au
import validation_utils as vu


# # Defining the Table

# In[ ]:


import neuron_utils as nru
import neuron
import trimesh_utils as tu
import numpy as np


# In[ ]:


import meshlab
meshlab.set_meshlab_port(current_port=None)


# # Creating the Auto Proofread Neuron Table

# In[ ]:


# minnie,schema = du.configure_minnie_vm()
# minnie.AutoProofreadValidationBorderNeurons.drop()
# minnie.AutoProofreadValidationBorder.drop()
# minnie.schema.external['decomposition'].delete(delete_external_files=True)


# In[ ]:


@schema
class AutoProofreadValidationBorderNeurons(dj.Manual):
    definition="""
    -> minnie.Decomposition() 
    ---
    decomposition: <decomposition>
    """


# In[ ]:


import numpy as np
import time

verbose = True
axon_version = 0
split_version = 2
@schema
class AutoProofreadValidationBorder(dj.Computed):
    definition="""
    -> minnie.Decomposition() 
    axon_version    : tinyint unsigned 
    parent_idx: smallint unsigned
    ---
    n_downstream=NULL: smallint unsigned
    web_size_faces=NULL : smallint unsigned
    web_size_volume=NULL : double
    web_size_skeleton=NULL : double
    web_size_ray_trace_percentile=NULL : double
    web_bbox_ratios_max=NULL: double
    web_bbox_ratios_min=NULL: double
    web_volume_ratio=NULL: double
    web_cdf=NULL: double
    parent_n_large_boutons=NULL : tinyint unsigned 
    parent_n_boutons=NULL : tinyint unsigned 
    parent_no_bouton_median=NULL : double
    parent_no_spine_median_mesh_center=NULL : double
    child_no_bouton_median_min =NULL: double
    child_no_bouton_median_diff_min=NULL : double
    child_no_spine_median_mesh_center_min=NULL :double
    child_no_spine_median_mesh_center_diff_min=NULL : double
    child_angle_min=NULL : double
    child_n_boutons_min=NULL : tinyint unsigned
    child_n_large_boutons_min=NULL : tinyint unsigned
    child_no_bouton_median_max =NULL: double
    child_no_bouton_median_diff_max=NULL: double
    child_no_spine_median_mesh_center_max=NULL : double
    child_no_spine_median_mesh_center_diff_max=NULL :double
    child_angle_max=NULL : double
    child_n_boutons_max =NULL: tinyint unsigned
    child_n_large_boutons_max=NULL : tinyint unsigned
    sibling_angles_min=NULL: double
    sibling_angles_max=NULL: double
    label: varchar(10)
    
    """

    #key_source = du.proofreading_stats_table(validation=True)
    key_source = (minnie.Decomposition() 
                  & (minnie.AutoProofreadValidationSegment() - minnie.AutoProofreadValidationSegmentExclude() ))

    

    def make(self,key):
        """
        Pseudocode:
        1) Pull down the neuron object
        2) Run the complete axon preprocessing on the neuron
        3) Run the borders attributes dictionary
        4) Save off the neuron object
        5) Write the Attribute records

        """
        print(f"\n\n\n---- Working on Neuron {key['segment_id']} ----")
        
        # 1) Pull Down All of the Neurons
        segment_id = key["segment_id"]
        
        whole_pass_time = time.time()
        neuron_objs,neuron_split_idxs = du.decomposition_with_spine_recalculation(segment_id)
        split_index = 0
#         key["split_index"] = split_index
#         key["split_version"] = split_version
        neuron_obj = neuron_objs[split_index]
        
        #2) Run the complete axon preprocessing on the neuron
        neuron_obj_with_web = au.complete_axon_processing(neuron_obj,
                                                 verbose=True)
        
        
        branch_attr = vu.neuron_to_border_branching_attributes(neuron_obj_with_web,
                                         plot_valid_border_branches=False,
                                          plot_invalid_border_branches = False,
                                          verbose=False
                                         )
        
        #3) Run the borders attributes dictionary
        branch_attr_keys = []
        for k in branch_attr:
            new_dict  = dict(key)
            new_dict.update(k)
            new_dict["axon_version"] = axon_version
            branch_attr_keys.append(new_dict)
            
        if verbose:
            print(f"\n\nlen(branch_attr_keys) = {len(branch_attr_keys)}")
            
            
        
        #4) Save the file in a certain location
        save_time = time.time()
        ret_file_path = neuron_obj_with_web.save_compressed_neuron(output_folder=str(du.get_decomposition_path()),
                                        file_name=f"{neuron_obj_with_web.segment_id}_validation_full_axon",
                                          return_file_path=True,
                                         export_mesh=False,
                                         suppress_output=False)

        ret_file_path_str = str(ret_file_path.absolute()) + ".pbz2"
        print(f"ret_file_path_str = {ret_file_path_str}")
        print(f"Save time = {time.time() - save_time}")
        
        n_dict = dict(key,
                     decomposition=ret_file_path_str)
        
        AutoProofreadValidationBorderNeurons.insert1(n_dict,skip_duplicates=True)
        
        
        
        
        #5) Write the Attribute records
        if len(branch_attr_keys)>0:
            AutoProofreadValidationBorder.insert(branch_attr_keys,skip_duplicates=True)

    

        print(f"\n\n ***------ Total time for {key['segment_id']} = {time.time() - whole_pass_time} ------ ***")
    


# # Running the Populate

# In[ ]:


dj.config["display.limit"] = 30
curr_table = (minnie.schema.jobs & "table_name='__auto_proofread_validation_border'")
(curr_table)#.delete()# & "status='error'")
#curr_table#.delete()
#(curr_table & "error_message = 'ValueError: need at least one array to concatenate'").delete()


# In[ ]:


import time
import neuron
pru = reload(pru)
nru = reload(nru)
neuron = reload(neuron)
import neuron_searching as ns
ns = reload(ns)
clu = reload(clu)
du = reload(du)
import axon_utils as au
au = reload(au)
import random
import skeleton_utils as sk
sk = reload(sk)
import trimesh_utils as tu
tu = reload(tu)

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 800))
print('Populate Started')
if not test_mode:
    AutoProofreadValidationBorder.populate(reserve_jobs=True, suppress_errors=True, order="random")
else:
    AutoProofreadValidationBorder.populate(reserve_jobs=True, suppress_errors=False, order="random")
print('Populate Done')

print(f"Total time for AutoProofreadValidationBorder populate = {time.time() - start_time}")

