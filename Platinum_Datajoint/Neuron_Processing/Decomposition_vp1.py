#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To Run the neuron preprocessing


"""


# In[1]:


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

# In[2]:


import minfig
import time
import numpy as np
#want to add in a wait for the connection part
random_sleep_sec = np.random.randint(0, 30)
print(f"Sleeping {random_sleep_sec} sec before conneting")
time.sleep(random_sleep_sec)
print("Done sleeping")

du.config_celii()
du.set_minnie65_config_segmentation(minfig)
du.print_minnie65_config_paths(minfig)

#configuring will include the adapters
minnie,schema = du.configure_minnie_vm()


# # Defining Our Table

# In[3]:


import neuron_utils as nru
import neuron
import trimesh_utils as tu
import numpy as np


# In[4]:


from soma_extraction_utils import *
import meshlab
meshlab.set_meshlab_port(current_port=None)


# In[5]:


#so that it will have the adapter defined
from datajoint_utils import *


# In[ ]:


#minnie.Decomposition.drop()
#schema.external['decomposition'].delete(delete_external_files=True)


# In[ ]:


import numpy as np
import time
decimation_version = 0
decimation_ratio = 0.25

@schema
class Decomposition(dj.Computed):
    definition="""
    -> minnie.Decimation.proj(decimation_version='version')
    ---
    decomposition: <decomposition>
    n_vertices           : int unsigned                 # number of vertices
    n_faces              : int unsigned                 # number of faces
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
    spine_volume_per_branch_eligible=NULL: double #total_spine_volume/n_spine_eligible_branche

    
    """

    key_source =  ((minnie.Decimation).proj(decimation_version='version') & 
                            "decimation_version=" + str(decimation_version) &
                       f"decimation_ratio={decimation_ratio}" &  (minnie.BaylorSegmentCentroid() & "multiplicity>0").proj())
    

    def make(self,key):
        """
        Pseudocode for process:

        1) Get the segment id from the key
        2) Get the decimated mesh
        3) Get the somas info
        4) Run the preprocessing
        5) Calculate all starter stats
        6) Save the file in a certain location
        7) Pass stats and file location to insert
        """
        #1) Get the segment id from the key
        segment_id = key["segment_id"]
        description = str(key['decimation_version']) + "_25"
        print(f"\n\n----- Working on {segment_id}-------")
        global_start = time.time()
        
        #2) Get the decimated mesh
        current_neuron_mesh = du.fetch_segment_id_mesh(segment_id)

        #3) Get the somas info *************************** Need to change this when actually run *******************
        somas = du.get_soma_mesh_list(segment_id) 
        print(f"somas = {somas}")
        #4) Run the preprocessing


        total_neuron_process_time = time.time()

        print(f"\n--- Beginning preprocessing of {segment_id}---")
        recovered_neuron = neuron.Neuron(
        mesh = current_neuron_mesh,
        somas = somas,
        segment_id=segment_id,
        description=description,
        suppress_preprocessing_print=False,
        suppress_output=False,
        calculate_spines=True,
        widths_to_calculate=["no_spine_median_mesh_center"]

                )

        print(f"\n\n\n---- Total preprocessing time = {time.time() - total_neuron_process_time}")


        #5) Don't have to do any of the processing anymore because will do in the neuron object
        stats_dict = recovered_neuron.neuron_stats()



        #6) Save the file in a certain location
        save_time = time.time()
        ret_file_path = recovered_neuron.save_compressed_neuron(output_folder=str(du.get_decomposition_path()),
                                          return_file_path=True,
                                         export_mesh=False,
                                         suppress_output=True)

        ret_file_path_str = str(ret_file_path.absolute()) + ".pbz2"
        print(f"Save time = {time.time() - save_time}")



        #7) Pass stats and file location to insert
        new_key = dict(key,
                       decomposition=ret_file_path_str,
                       n_vertices=len(current_neuron_mesh.vertices),
                       n_faces=len(current_neuron_mesh.faces)
                      )
        new_key.update(stats_dict)

        self.insert1(new_key, allow_direct_insert=True, skip_duplicates=True)

        print(f"\n\n ------ Total time for {segment_id} = {time.time() - global_start} ------")
    


# # Running The Populate

# In[6]:


#(minnie.schema.jobs & "table_name='__decomposition'").delete()
#((schema.jobs & "table_name = '__decomposition'") & "timestamp>'2020-11-16 00:26:00'").delete()


# In[ ]:


import time
import random

start_time = time.time()
time.sleep(random.randint(0, 900))
print('Populate Started')
Decomposition.populate(reserve_jobs=True, suppress_errors=True, order='random')
print('Populate Done')

print(f"Total time for Decomposition populate = {time.time() - start_time}")

