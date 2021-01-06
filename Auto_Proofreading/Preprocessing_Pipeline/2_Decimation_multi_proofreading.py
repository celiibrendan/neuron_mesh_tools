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

import datajoint_utils as du
from importlib import reload


# In[ ]:


test_mode = False


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


# # Getting the list of neurons to decompose for the mutli soma testing

# In[ ]:


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

# In[ ]:


import neuron_utils as nru
import neuron
import trimesh_utils as tu
import numpy as np


# In[ ]:


import meshlab
meshlab.set_meshlab_port(current_port=None)
temporary_folder = 'decimation_temp'
meshlab_scripts = {}


# In[ ]:


#so that it will have the adapter defined
from datajoint_utils import *


# In[ ]:


import numpy as np
import time
decimation_version = 0
decimation_ratio = 0.25

from minfig.minnie65_config import external_decimated_mesh_path

@schema
class Decimation(dj.Computed):
#     definition = minnie.Decimation.describe(printout=False)
    key_source = minnie.Mesh.proj() * (minnie.DecimationConfig & 'decimation_ratio=0.25') & minnie.MultiSomaProofread2.proj()

    # Creates hf file at the proper location, returns the filepath of the newly created file
    @classmethod
    def make_file(cls, segment_id, version, decimation_ratio, vertices, faces):
        """Creates hf file at the proper location, returns the filepath of the newly created file"""

        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3

        filename = f'{segment_id}_{version}_{int(decimation_ratio*100):02}.h5'
        filepath = os.path.join(external_decimated_mesh_path, filename)
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('segment_id', data=segment_id)
            hf.create_dataset('version', data=version)
            hf.create_dataset('decimation_ratio', data=float(decimation_ratio))
            hf.create_dataset('vertices', data=vertices)
            hf.create_dataset('faces', data=faces)

        return filepath

    @classmethod
    def make_entry(cls, segment_id, version, decimation_ratio, vertices, faces):
        key = dict(
            segment_id=segment_id,
            version=version,
            decimation_ratio=decimation_ratio,
            n_vertices=len(vertices),
            n_faces=len(faces)
        )

        filepath = cls.make_file(segment_id, version, decimation_ratio, vertices, faces)

        cls.insert1(dict(key, mesh=filepath), allow_direct_insert=True)

    

    def make(self, key):
        print(key)
        mesh = (minnie.Mesh & key).fetch1('mesh')
        segment_id = key['segment_id']
        version = key['version']
        decimation_ratio = key['decimation_ratio']
        print(f"Mesh size: n_vertices = {len(mesh.vertices)}, n_faces = {len(mesh.faces)}")

        if decimation_ratio not in meshlab_scripts:
            meshlab_scripts[decimation_ratio] = meshlab.Decimator(decimation_ratio, temporary_folder, overwrite=False)
        mls_func = meshlab_scripts[decimation_ratio]

        try:
            expected_filepath = os.path.join(external_decimated_mesh_path, f'{segment_id}_{version}.h5')
            if not os.path.isfile(expected_filepath):
                new_mesh, _path = mls_func(mesh.vertices, mesh.faces, segment_id)
                new_vertices, new_faces = new_mesh.vertices, new_mesh.faces

                self.make_entry(
                    segment_id=segment_id,
                    version=version,
                    decimation_ratio=decimation_ratio,
                    vertices=new_vertices,
                    faces=new_faces,
                    )
            else:
                print('File already exists.')
                with h5py.File(expected_filepath, 'r') as hf:
                    vertices = hf['vertices'][()].astype(np.float64)
                    faces = hf['faces'][()].reshape(-1, 3).astype(np.uint32)
                self.insert1(dict(key, n_vertices=len(vertices), n_faces=len(faces), mesh=expected_filepath), allow_direct_insert=True)
        except Exception as e:
            minnie.DecimationError.insert1(dict(key, log=str(e)))
            print(e)
            raise e
    


# # Running the Populate

# In[ ]:


curr_table = (minnie.schema.jobs & "table_name='__decimation'")
#curr_table.delete()
#curr_table.delete()


# In[ ]:


# import pandas as pd
# key_hash,error_message = curr_table.fetch("key_hash","error_message")

# df = pd.DataFrame.from_dict([dict(key_hash = k,error_message = m) for k,m in zip(key_hash,error_message)])
# df
# #df.columns = ["error","key_hash"]
# key_hashes_to_delete = df[df["error_message"].str.contains("OSError")]["key_hash"].to_numpy()

# (curr_table & [dict(key_hash=k) for k in key_hashes_to_delete]).delete()


# In[ ]:


import time
import random

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 800))
print('Populate Started')
if not test_mode:
    Decimation.populate(reserve_jobs=True, suppress_errors=True)
else:
    Decimation.populate(reserve_jobs=True, suppress_errors=False)
print('Populate Done')

print(f"Total time for Decimation populate = {time.time() - start_time}")


# In[ ]:


debug


# In[ ]:




