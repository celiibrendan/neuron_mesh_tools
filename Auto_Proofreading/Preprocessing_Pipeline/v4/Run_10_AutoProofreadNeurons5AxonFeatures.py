#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To compute different features of an axon that can be used to pick good axons

Brainstorming: 
2) The volume spanned
3) x/y/z min/max
4) Number of branches
5) average skeletal length per branch
6) median skeletal length per branch
7) axon length
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


# # Creating the Auto Proofread Neuron Table

# In[ ]:


import numpy as np
import time
import classification_utils as clu
import axon_utils as au

@schema
class AutoProofreadNeurons5AxonFeatures(dj.Computed):
    definition="""
    -> minnie.AutoProofreadNeurons5()
    ---
    # -------- Important Excitatory Inhibitory Classfication ------- #
    volume: double #volume of the oriented bounding box of axon (divided by 10^14)
    
    axon_length: double  # length (in um) of the classified axon skeleton
    axon_branch_length_median: double  # length (in um) of the classified axon skeleton
    axon_branch_length_mean: double  # length (in um) of the classified axon skeleton
    
    # number of branches in the axon
    n_branches: int unsigned  
    n_short_branches:  int unsigned
    n_long_branches:  int unsigned
    n_medium_branches:  int unsigned
    
    #bounding box features
    bbox_x_min: double 
    bbox_y_min: double 
    bbox_z_min: double 
    bbox_x_max: double 
    bbox_y_max: double 
    bbox_z_max: double 
    
    bbox_x_min_soma_relative: double 
    bbox_y_min_soma_relative: double 
    bbox_z_min_soma_relative: double 
    bbox_x_max_soma_relative: double 
    bbox_y_max_soma_relative: double 
    bbox_z_max_soma_relative: double 
    
    """
    
    key_source = minnie.AutoProofreadNeurons5() & "axon_length > 0" & "spine_category='densely_spined'"
    

    def make(self,key):
        """
        Pseudocode:
        1) Pull Down All of the Neurons
        2) Get the nucleus centers and the original mesh

        """
        
        # 1) Pull Down All of the Neurons
        segment_id = key["segment_id"]
        split_index = key["split_index"]
        
        whole_pass_time = time.time()
        
        axon_dict = au.axon_stats_from_proof_axon_skeleton(segment_id,
                                              split_index)
        dict_to_write = dict(key)
        dict_to_write.update(axon_dict)
        
        self.insert1(dict_to_write,skip_duplicates=True,allow_direct_insert=True)
            

        print(f"\n\n ***------ Total time for {key['segment_id']} = {time.time() - whole_pass_time} ------ ***")


# # Running the Populate

# In[ ]:


curr_table = (minnie.schema.jobs & "table_name='__auto_proofread_neurons5_axon_features'")
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
    AutoProofreadNeurons5AxonFeatures.populate(reserve_jobs=True, suppress_errors=True, order="random")
else:
    AutoProofreadNeurons5AxonFeatures.populate(reserve_jobs=True, suppress_errors=False, order="random")
print('Populate Done')

print(f"Total time for AutoProofreadNeurons5AxonFeatures populate = {time.time() - start_time}")


# In[ ]:




