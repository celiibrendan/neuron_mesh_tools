#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
To Run the Error Labeling Pipeline


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


import error_detection as ed
ed = reload(ed)


# In[ ]:


import numpy as np
import time
decimation_version = 0
decimation_ratio = 0.25

@schema
class AutoProofreadLabels(dj.Computed):
    definition="""
    -> minnie.Decomposition
    ---
    n_face_errors : int #the number of faces that were errored out
    face_idx_for_error : longblob #the face indices for the errors computed
    n_synapses: smallint unsigned #total number of synpases
    n_errored_synapses: smallint unsigned #the number of synapses
    """
    
    
    
    key_source = (minnie.Decomposition() & "n_somas = 1" & "n_faces>500000")
                  
    def make(self,key):
        global_start = time.time()
        segment_id = key["segment_id"]
        verbose = True
        
        print(f"\n\n----- Working on {segment_id}-------")
        whole_pass_time = time.time()
        
        #new method that checks if the information exists in the error table and if not then 
        error_table = (minnie.DecompositionErrorLabels() & dict(segment_id=segment_id))
        if len(error_table)>0:
            print("using quick fetch")
            current_mesh = du.fetch_segment_id_mesh(segment_id)
            returned_error_faces = error_table.fetch1("face_idx_for_error")
            
        else:
            neuron_obj = (minnie.Decomposition() & key).fetch1("decomposition")

            returned_error_faces = ed.error_faces_by_axons(neuron_obj,verbose=True,visualize_errors_at_end=False)
            current_mesh = neuron_obj.mesh
            
        #------- Doing the synapse Exclusion Writing ---------- #
        data_to_write_new,n_synapses,n_errored_synapses = ed.get_error_synapse_inserts(current_mesh,
                                                                                       segment_id,
                                                                                       returned_error_faces,minnie=minnie,
                                                         return_synapse_stats=True,
                                                         verbose=True)
        
        if len(data_to_write_new)>0:
            print("Preparing to write errored synapses")
            minnie.SynapseExclude.insert(data_to_write_new,skip_duplicates=True)
            
        #------- Doing the Label Writing ---------- #
        new_key = dict(key,
                       n_face_errors = len(returned_error_faces),
                       face_idx_for_error = returned_error_faces,
                        n_synapses=n_synapses,
                        n_errored_synapses=n_errored_synapses)
        
        
        self.insert1(new_key, allow_direct_insert=True, skip_duplicates=True)
        
        print(f"\n\n ------ Total time for {segment_id} = {time.time() - global_start} ------")
        


# In[ ]:


#(schema.jobs & "table_name='__decomposition_error_labels'").delete()
#minnie.SynapseExclude.delete()
#minnie.DecompositionErrorLabels.delete()


# In[ ]:


import time
import random

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 900))
print('Populate Started')
if test_mode:
    AutoProofreadLabels.populate(reserve_jobs=True, suppress_errors=False)
else:
    AutoProofreadLabels.populate(reserve_jobs=True, suppress_errors=True)
print('Populate Done')

print(f"Total time for DecompositionErrorLabels populate = {time.time() - start_time}")

