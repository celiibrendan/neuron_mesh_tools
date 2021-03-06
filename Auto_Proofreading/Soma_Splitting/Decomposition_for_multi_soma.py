#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To decompose the multi-somas for splitting
using the new decomposition method



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


# In[2]:


test_mode = False


# In[11]:


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


# # Getting the list of neurons to decompose for the mutli soma testing

# In[4]:


# import pandas as pd
# soma_soma_table = pd.read_csv("Minnie65 core proofreading - Soma-Soma.csv")
# no_header = soma_soma_table.iloc[1:]
# multi_soma_ids_str = no_header["Dendrites"].to_numpy()
# multi_soma_ids = multi_soma_ids_str[~np.isnan(multi_soma_ids_str.astype("float"))].astype("int")

# @schema
# class MultiSomaProofread(dj.Manual):
#     definition="""
#     segment_id : bigint unsigned  #segment id for those to be decimated
#     """
    
# dict_of_seg = [dict(segment_id=k) for k in multi_soma_ids]
# minnie.MultiSomaProofread.insert(dict_of_seg,skip_duplicates=True)
# MultiSomaProofread()


# # Defining the Table

# In[5]:


import neuron_utils as nru
import neuron
import trimesh_utils as tu
import numpy as np


# In[6]:


import meshlab
meshlab.set_meshlab_port(current_port=None)


# In[7]:


#so that it will have the adapter defined
from datajoint_utils import *


# In[13]:


decimation_version = 0
decimation_ratio = 0.25
key_source = (minnie.Decimation().proj(decimation_version='version')  & 
              dict(decimation_version=decimation_version,decimation_ratio=decimation_ratio)  
              & minnie.MultiSomaProofread() & (dj.U("segment_id") & (minnie.BaylorSegmentCentroid() & "multiplicity=2").proj()))
key_source


# In[14]:


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
    spine_volume_per_branch_eligible=NULL: double #total_spine_volume/n_spine_eligible_branches
    
    run_time=NULL : double                   # the amount of time to run (seconds)

    
    """

#     key_source =  ((minnie.Decimation).proj(decimation_version='version') & 
#                             "decimation_version=" + str(decimation_version) &
#                        f"decimation_ratio={decimation_ratio}" &  (minnie.BaylorSegmentCentroid() & "multiplicity>0" & "segment_id=864691136309663834").proj())
#     key_source = (minnie.Decimation() & "n_faces>500000").proj(decimation_version='version') & (minnie.BaylorSegmentCentroid() & "multiplicity=1").proj()

    
    key_source = (minnie.Decimation().proj(decimation_version='version')  & 
                  dict(decimation_version=decimation_version,decimation_ratio=decimation_ratio)  
                  & minnie.MultiSomaProofread() & (dj.U("segment_id") & (minnie.BaylorSegmentCentroid() & "multiplicity=2").proj()))

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
        whole_pass_time = time.time()
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
                       n_faces=len(current_neuron_mesh.faces),
                       run_time=np.round(time.time() - whole_pass_time,4)
                      )
        new_key.update(stats_dict)

        self.insert1(new_key, allow_direct_insert=True, skip_duplicates=True)

        print(f"\n\n ------ Total time for {segment_id} = {time.time() - global_start} ------")
    


# # Running the Populate

# In[22]:


curr_table = (minnie.schema.jobs & "table_name='__decomposition'")
#curr_table.delete()
#(curr_table & "error_message = 'ValueError: need at least one array to concatenate'").delete()


# In[19]:


# import pandas as pd
# key_hash,error_message = curr_table.fetch("key_hash","error_message")

# df = pd.DataFrame.from_dict([dict(key_hash = k,error_message = m) for k,m in zip(key_hash,error_message)])
# df
# #df.columns = ["error","key_hash"]
# key_hashes_to_delete = df[df["error_message"].str.contains("OSError")]["key_hash"].to_numpy()

# (curr_table & [dict(key_hash=k) for k in key_hashes_to_delete]).delete()


# In[20]:


import time
import random
import compartment_utils as cu
cu = reload(cu)
import preprocessing_vp2 as pre
pre = reload(pre)

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 800))
print('Populate Started')
if not test_mode:
    Decomposition.populate(reserve_jobs=True, suppress_errors=True)
else:
    Decomposition.populate(reserve_jobs=True, suppress_errors=False)
print('Populate Done')

print(f"Total time for DecompositionMultiSoma populate = {time.time() - start_time}")


# In[ ]:




