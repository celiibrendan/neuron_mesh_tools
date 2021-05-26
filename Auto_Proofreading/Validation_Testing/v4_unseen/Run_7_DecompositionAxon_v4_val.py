#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Purpose: To Create the table that
will store the neuron objects that have finer
axon preprocessing

"""


# In[2]:


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


# In[3]:


#so that it will have the adapter defined
from datajoint_utils import *


# In[4]:


test_mode = False


# # Debugging the contains method

# In[5]:


import system_utils as su


# In[6]:


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


# In[7]:


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

# In[8]:


import meshlab
meshlab.set_meshlab_port(current_port=None)


# # Proofreading Version

# In[9]:


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
         [5,"filtered away floating pieces near soma for stitching"]]

dict_to_write = [dict(axon_version=k,description=v) for k,v in versions]
DecompositonAxonVersion.insert(dict_to_write,skip_duplicates=True)

DecompositonAxonVersion()


# In[38]:


# minnie,schema = du.configure_minnie_vm()
# minnie.DecompositionAxon.delete()
#minnie.schema.external['decomposition'].delete(delete_external_files=True)


# In[11]:


import numpy as np
import time
import classification_utils as clu
import proofreading_utils as pru
import axon_utils as au

axon_version = au.axon_version

verbose = True

@schema
class DecompositionAxon(dj.Computed):
    definition="""
    -> minnie.Decomposition()
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    -> minnie.DecompositonAxonVersion()             # the version of code used for this cell typing classification
    ---
    decomposition        : <decomposition> # saved neuron object with high fidelity axon
    axon_length: double  # length (in um) of the classified axon skeleton
    run_time=NULL : double                   # the amount of time to run (seconds)
    """
                             
    
    #key_source = minnie.Decomposition() & minnie.NucleiSegmentsRun2() & "segment_id=864691136540183458"
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
        neuron_objs,neuron_split_idxs = du.decomposition_with_spine_recalculation(segment_id)

        if verbose:
            print(f"Number of Neurons found ={len(neuron_objs)}")

        #For each neuron:
        dict_to_write = []
        for split_index,neuron_obj in zip(neuron_split_idxs,neuron_objs):
            
            if verbose:
                print(f"--> Working on Split Index {split_index} -----")
                
            st = time.time()
            #Run the Axon Decomposition
            neuron_obj_with_web = au.complete_axon_processing(neuron_obj,
                                     verbose=True)
            
            save_time = time.time()
            ret_file_path = neuron_obj_with_web.save_compressed_neuron(
                                            output_folder=str(du.get_decomposition_path()),
                                            #output_folder = "./",
            file_name=f"{neuron_obj_with_web.segment_id}_{split_index}_split_axon_v{au.axon_version}",
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
             axon_length=neuron_obj_with_web.axon_length,
              run_time = np.round(time.time() - st,2)
             )
            
            dict_to_write.append(n_dict)
        
        #write the
        self.insert(dict_to_write,skip_duplicates=True,allow_direct_insert=True)

        print(f"\n\n ***------ Total time for {key['segment_id']} = {time.time() - whole_pass_time} ------ ***")


# # Running the Populate

# In[37]:


curr_table = (minnie.schema.jobs & "table_name='__decomposition_axon'")
(curr_table)#.delete()# & "status='error'")
#curr_table.delete()
#(curr_table & "status='error'") #& "error_message='IndexError: list index out of range'"


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
    DecompositionAxon.populate(reserve_jobs=True, suppress_errors=True, order="random")
else:
    DecompositionAxon.populate(reserve_jobs=True, suppress_errors=False,)# order="random")
print('Populate Done')

print(f"Total time for DecompositionAxon populate = {time.time() - start_time}")

