#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
To Run the Error Labeling Pipeline


"""


# In[2]:


import numpy as np
import datajoint as dj
import trimesh
from tqdm.notebook import tqdm
from pathlib import Path

from os import sys
sys.path.append("/meshAfterParty/")


from importlib import reload


# # configuring the virtual module

# In[3]:


test_mode = False


# In[4]:


import minfig
import time
import numpy as np
#want to add in a wait for the connection part
random_sleep_sec = np.random.randint(0, 400)
print(f"Sleeping {random_sleep_sec} sec before conneting")


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

# In[5]:


import neuron_utils as nru
import neuron
import trimesh_utils as tu
import numpy as np


# In[6]:


#so that it will have the adapter defined
from datajoint_utils import *


# In[7]:


import error_detection as ed


# In[8]:


import numpy as np
import time
decimation_version = 0
decimation_ratio = 0.25

@schema
class DecompositionErrorLabels(dj.Computed):
    definition="""
    -> minnie.Decomposition
    ---
    n_face_errors : int #the number of faces that were errored out
    face_idx_for_error : longblob #the face indices for the errors computed
    """
    
    
    
    key_source = (minnie.Decomposition() & "n_somas = 1" & "n_faces>500000")
                  
    def make(self,key):
        global_start = time.time()
        segment_id = key["segment_id"]
        verbose = True
        
        print(f"\n\n----- Working on {segment_id}-------")
        whole_pass_time = time.time()
        
        neuron_obj = (minnie.Decomposition() & key).fetch1("decomposition")
        
        returned_error_faces = ed.error_faces_by_axons(neuron_obj,verbose=True,visualize_errors_at_end=False)
        
        #------- Doing the synapse Exclusion Writing ---------- #
        
        
        success_flag = False
        for i in range(10):
            try:
                data_to_write_new = ed.get_error_synapse_inserts(neuron_obj,returned_error_faces,minnie=minnie,verbose=True)

            except:
                print("Locked out trying to get data trying agin in 30 seconds")
                time.sleep(30)
            else:
                success_flag = True


            if success_flag:
                print("successfully configured minnie")
                break
        
        if not success_flag:
            raise Exception("still didn't get data")
            
        if len(data_to_write_new)>0:
            print("Preparing to write errored synapses")
            minnie.SynapseExclude.insert(data_to_write_new,skip_duplicates=True)
            
        #------- Doing the Label Writing ---------- #
        new_key = dict(key,
                       n_face_errors = len(returned_error_faces),
                       face_idx_for_error = returned_error_faces)
        
        
        self.insert1(new_key, allow_direct_insert=True, skip_duplicates=True)
        
        print(f"\n\n ------ Total time for {segment_id} = {time.time() - global_start} ------")
        


# In[9]:


#(schema.jobs & "table_name='__decomposition_error_labels'").delete()
#minnie.SynapseExclude.delete()
#minnie.DecompositionErrorLabels.delete()


# In[10]:


import time
import random

start_time = time.time()
time.sleep(random.randint(0, 900))

DecompositionErrorLabels.populate(reserve_jobs=True, suppress_errors=True, order='random')
print('Populate Done')

print(f"Total time for DecompositionErrorLabels populate = {time.time() - start_time}")


# In[ ]:




