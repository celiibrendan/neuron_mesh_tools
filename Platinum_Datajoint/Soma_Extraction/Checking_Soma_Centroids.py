#!/usr/bin/env python
# coding: utf-8

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
import soma_extraction_utils as sm


# # debugging

# In[2]:


import system_utils as su
new_submesh = su.decompress_pickle("./new_submesh")


# In[3]:


new_submesh.split(only_watertight=False,repair=False)


# In[4]:


import minfig
du.set_minnie65_config_segmentation(minfig)
du.print_minnie65_config_paths(minfig)


# In[5]:


minnie = minfig.configure_minnie(return_virtual_module=True)

# Old way of getting access to the virtual modules
# m65 = dj.create_virtual_module('minnie', 'microns_minnie65_02')

#New way of getting access to module
import datajoint as dj
from minfig import adapter_objects # included with wildcard imports
minnie = dj.create_virtual_module('minnie', 'microns_minnie65_02', add_objects=adapter_objects)

schema = dj.schema("microns_minnie65_02")
dj.config["enable_python_native_blobs"] = True
#(schema.jobs & "table_name='__baylor_segment_centroid_seg3'").delete()


# # Fetching and Visualizing Meshes

# In[6]:


# def get_decimated_mesh(seg_id,decimation_ratio=0.25):
#     key = dict(segment_id=seg_id,decimation_ratio=decimation_ratio)
#     new_mesh = (minnie.Decimation() & key).fetch1("mesh")
#     current_mesh_verts,current_mesh_faces = new_mesh.vertices,new_mesh.faces
#     return trimesh.Trimesh(vertices=current_mesh_verts,faces=current_mesh_faces)

# def get_seg_extracted_somas(seg_id):
#     key = dict(segment_id=seg_id)  
#     soma_vertices, soma_faces = (minnie.BaylorSegmentCentroid() & key).fetch("soma_vertices","soma_faces")
#     return [trimesh.Trimesh(vertices=v,faces=f) for v,f in zip(soma_vertices, soma_faces)]
# def get_soma_mesh_list(seg_id):
#     key = dict(segment_id=seg_id)  
#     soma_vertices, soma_faces,soma_run_time,soma_sdf = (minnie.BaylorSegmentCentroid() & key).fetch("soma_vertices","soma_faces","run_time","sdf")
#     s_meshes = [trimesh.Trimesh(vertices=v,faces=f) for v,f in zip(soma_vertices, soma_faces)]
#     s_times = list(soma_run_time)
#     s_sdfs = list(soma_sdf)
#     return [s_meshes,s_times,s_sdfs]


# In[7]:


# multi_soma_seg_ids = np.unique(multi_soma_seg_ids)
# seg_id_idx = -2
# seg_id = multi_soma_seg_ids[seg_id_idx]

# dec_mesh = get_decimated_mesh(seg_id)
# curr_soma_meshes = get_seg_extracted_somas(seg_id)
# curr_soma_mesh_list = get_soma_mesh_list(seg_id)

# import skeleton_utils as sk
# sk.graph_skeleton_and_mesh(main_mesh_verts=dec_mesh.vertices,
#                            main_mesh_faces=dec_mesh.faces,
#                         other_meshes=curr_soma_meshes,
#                           other_meshes_colors="red")


# # Exploring the New Query and Seeing if matches up with nucleus id:

# In[8]:


decimation_version = 0.25

valid_segment_ids_with_nucleus_id = dj.U("segment_id") & (minnie.NucleusID() & "segment_id>0")
segments = (minnie.Mesh() & f'n_vertices  > {np.str(np.round(np.exp(12)).astype(np.int))}' & f'n_vertices  < {np.str(np.round(np.exp(15)).astype(np.int))}')
key_source =  minnie.Decimation.proj(decimation_version='version') & segments.proj() & f"decimation_version={decimation_version}" & valid_segment_ids_with_nucleus_id
key_source


# In[9]:


du = reload(du)
seg_id = 864691135635239593
du.plot_decimated_mesh_with_somas(seg_id)

#error_poisson_somas = du.get_seg_extracted_somas(seg_id)
error_mesh = du.get_decimated_mesh(seg_id)

# error_backtrack_somas = sm.original_mesh_soma(
#                     mesh = error_mesh,
#                     soma_meshes=error_poisson_somas,
#                     sig_th_initial_split=15)


# In[ ]:


import trimesh_utils as tu
sm = reload(sm)
tu = reload(tu)

soma_data = sm.extract_soma_center(seg_id,
                      error_mesh.vertices,
                      error_mesh.faces)


# # -- getting neurons to help test on -- 

# In[ ]:


# sm = reload(sm)


# In[ ]:


# import neuron_utils as nru
# filepath = "/notebooks/test_neurons/meshafterparty_processed/12345_double_soma_meshafterparty_fixed_connectors_and_spines"
# double_neuron = nru.decompress_neuron(
#     filepath=filepath,
#     original_mesh=filepath,
# )


# In[ ]:


# sm = reload(sm)
# double_neuron_poisson_somas = sm.extract_soma_center(double_neuron.segment_id,
#                                                     double_neuron.mesh.vertices,
#                                                     double_neuron.mesh.faces)
# d_neuron_poisson_somas,_ , _ = double_neuron_poisson_somas
# d_neuron_backtrack_somas,_,_ = double_neuron.get_somas()


# # -- Getting other meshes to try -- 

# In[ ]:


# exc_1 = tu.load_mesh_no_processing("/notebooks/test_neurons/spine_detection/95442489112204099_excitatory_7.off")
# soma_data = sm.extract_soma_center(1234,exc_1.vertices,exc_1.faces)


# In[ ]:


# exc_1_poisson_somas,_,_ = soma_data
# exc_1_backtrack_somas = sm.original_mesh_soma(
#                     mesh = exc_1,
#                     soma_meshes=exc_1_poisson_somas,
#                     sig_th_initial_split=15)


# In[ ]:


# exc_1_poisson_somas,exc_1_backtrack_somas


# In[ ]:


# sk.graph_skeleton_and_mesh(other_meshes=exc_1_poisson_somas+exc_1_backtrack_somas,
#                           other_meshes_colors=["black","red"])


# # --- Checking Neurons that should be processed -- 

# In[ ]:





# In[ ]:


# valid_segment_ids_with_nucleus_id = dj.U("segment_id") & (minnie.NucleusID() & "segment_id>0")


# In[ ]:


# len(minnie.NucleusID() & all_nucleus_id_segs)


# In[ ]:


# len(minnie.NucleusID() & "segment_id>0")


# In[ ]:


# minnie.NucleusCorePostsyn() & "n_soma = 2"# robust table 


# In[ ]:





# In[ ]:





# # Apply a check that looks and sees if has a border that is too big

# In[ ]:




