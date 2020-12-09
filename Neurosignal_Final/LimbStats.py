#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To Run the processing pipeline that
will extract statistics about the branching pattern 
and angles of branching in order to later cluster


"""


# In[ ]:


import numpy as np
import datajoint as dj
import trimesh
from tqdm.notebook import tqdm
from pathlib import Path

from os import sys
sys.path.append("/meshAfterParty/")

from importlib import reload


# # configuring the virtual module

# In[ ]:


test_mode = False


# In[ ]:


import minfig
import time
import numpy as np
#want to add in a wait for the connection part
random_sleep_sec = np.random.randint(0, 400)
print(f"Sleeping {random_sleep_sec} sec before conneting")

if not test_mode:
    time.sleep(random_sleep_sec)

print("Done sleeping")
import datajoint_utils as du
du.config_celii()
du.set_minnie65_config_segmentation(minfig)
du.print_minnie65_config_paths(minfig)

#configuring will include the adapters
success_flag = False
for i in range(10):
    try:
        minnie,schema = du.configure_minnie_vm()
        
    except:
        print("Locked out trying agin in 30 seconds")
        time.sleep(30)
    else:
        success_flag = True
        
        
    if success_flag:
        print("successfully configured minnie")
        break


# # Defining Our Table

# In[ ]:


import neuron_utils as nru
import neuron
import trimesh_utils as tu
import numpy as np


# In[ ]:


#so that it will have the adapter defined
from datajoint_utils import *


# In[ ]:


import neuron_statistics as n_st


# In[ ]:


import numpy as np
import time
decimation_version = 0
decimation_ratio = 0.25

@schema
class LimbStats(dj.Computed):
    definition="""
    -> minnie.Decomposition
    limb_idx : smallint unsigned #the limb id path was taken from
    path_idx : smallint unsigned #path identifier
    ---
    soma_angle=NULL: double
    n0_width_median_mesh_center=NULL: double
    n0_width_no_spine_median_mesh_center=NULL: double
    n0_n_spines=NULL:int
    n0_total_spine_volume=NULL: double
    n0_spine_volume_median=NULL: double
    n0_spine_volume_density=NULL: double
    n0_skeletal_length=NULL: double
    n0_parent_angle=NULL: double
    n0_sibling_angle=NULL: double
    n1_width_median_mesh_center=NULL: double
    n1_width_no_spine_median_mesh_center=NULL: double
    n1_n_spines=NULL:int
    n1_total_spine_volume=NULL: double
    n1_spine_volume_median=NULL: double
    n1_spine_volume_density=NULL: double
    n1_skeletal_length=NULL: double
    n1_parent_angle=NULL: double
    n1_sibling_angle=NULL: double
    """
    
    key_source = (minnie.Decomposition() & "n_error_limbs = 0" & "n_limbs > 4" & "n_somas=1" & "n_faces>500000")
    key_source
    
    def make(self,key):
        global_start = time.time()
        segment_id = key["segment_id"]
        verbose = True
        
        print(f"\n\n----- Working on {segment_id}-------")
        whole_pass_time = time.time()
        
        neuron_obj = (minnie.Decomposition() & key).fetch1("decomposition")
    
        dj_inserts = n_st.neuron_path_analysis(neuron_obj,
                                      plot_paths=False,
                                              verbose=False)
        
        #adding the key to the dictionaries to be inserted
        for k in dj_inserts:
            k.update(key)
        
        if len(dj_inserts)>0:
            print(f"Inserting {len(dj_inserts)} paths")
            LimbStats.insert(dj_inserts,allow_direct_insert=True, skip_duplicates=True)
        else:
            if verbose:
                print(f"Skipping inserts because none were present")


# In[ ]:


#(minnie.schema.jobs & "table_name='__limb_stats'").delete()
#((schema.jobs & "table_name = '__decomposition'") & "timestamp>'2020-11-16 00:26:00'").delete()


# In[ ]:


import time
import random

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 900))
print('Populate Started')
if test_mode:
    LimbStats.populate(reserve_jobs=True, suppress_errors=False)
else:
    LimbStats.populate(reserve_jobs=True, suppress_errors=True)
print('Populate Done')

print(f"Total time for LimbStats populate = {time.time() - start_time}")


# In[ ]:




