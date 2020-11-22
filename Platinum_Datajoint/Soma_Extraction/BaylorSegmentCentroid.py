#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To run the soma finder on the 
new mesh segmentation

"""


# # Modules for Datajoint

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


import minfig
import time
import numpy as np
#want to add in a wait for the connection part
random_sleep_sec = np.random.randint(0, 200)
print(f"Sleeping {random_sleep_sec} sec before conneting")
time.sleep(random_sleep_sec)
print("Done sleeping")

du.config_celii()
du.set_minnie65_config_segmentation(minfig)
du.print_minnie65_config_paths(minfig)

#configuring will include the adapters
minnie,schema = du.configure_minnie_vm()


# # Modules for Soma Extraction

# In[ ]:


from soma_extraction_utils import *


# In[ ]:


import meshlab
meshlab.set_meshlab_port(current_port=None)


# # The table that will do the soma extraction

# In[ ]:


#so that it will have the adapter defined
from datajoint_utils import *
from minfig.adapters import *
import trimesh_utils as tu


# In[ ]:


decimation_version = 0
decimation_ratio = 0.25
verts_min = 10000



@schema
class BaylorSegmentCentroid(dj.Computed):
    definition="""
    -> minnie.Decimation.proj(decimation_version='version')
    soma_index : tinyint unsigned #index given to this soma to account for multiple somas in one base semgnet
    ---
    centroid_x=NULL           : int unsigned                 # (EM voxels)
    centroid_y=NULL           : int unsigned                 # (EM voxels)
    centroid_z=NULL           : int unsigned                 # (EM voxels)
    n_vertices=NULL           : bigint                 #number of vertices
    n_faces=NULL            : bigint                  #number of faces
    mesh: <somas>  #datajoint adapter to get the somas mesh objects
    multiplicity=NULL         : tinyint unsigned             # the number of somas found for this base segment
    sdf=NULL                  : double                       # sdf width value for the soma
    max_side_ratio=NULL       : double                       # the maximum of the side length ratios used for check if soma
    bbox_volume_ratio=NULL    : double                       # ratio of bbox (axis aligned) volume to mesh volume to use for check if soma
    max_hole_length=NULL      : double                    #euclidean distance of the maximum hole size
    run_time=NULL : double                   # the amount of time to run (seconds)

    """
    key_source =  ((minnie.Decimation & f"n_vertices > {verts_min}").proj(decimation_version='version') & 
                        "decimation_version=" + str(decimation_version) &
                   f"decimation_ratio={decimation_ratio}" & minnie.SegToDecimateFromNuclei())

     

    def make(self,key):
        """
        Pseudocode: 
        1) Compute all of the
        2) Save the mesh as an h5 py file
        3) Store the saved path as the decomposition part of the dictionary and erase the vertices and faces
        4) Insert
        
        
        """
        
        #get the mesh data
        print(f"\n\n\n---- Working on {key['segment_id']} ----")
        print(key)
        new_mesh = (minnie.Decimation() & key).fetch1("mesh")
        current_mesh_verts,current_mesh_faces = new_mesh.vertices,new_mesh.faces

        segment_id = key["segment_id"]

        (total_soma_list, 
         run_time, 
         total_soma_list_sdf) = extract_soma_center(
                            segment_id,
                            current_mesh_verts,
                            current_mesh_faces,
        )
        
        
        print(f"Run time was {run_time} \n    total_soma_list = {total_soma_list}"
             f"\n    with sdf values = {total_soma_list_sdf}")
        
        #check if soma list is empty and did not find soma
        if len(total_soma_list) <= 0:
            print("There were no somas found for this mesh so just writing empty data")
            

            returned_file_path = tu.write_h5_file(
                                                vertices=np.array([]),
                                                  faces=np.array([]),
                                                  segment_id=segment_id,
                                                  filename = f'{segment_id}_0.h5',
                                                    filepath=str(du.get_somas_path())
                                                 )

            
            
            insert_dict = dict(key,
                              soma_index=0,
                              centroid_x=None,
                               centroid_y=None,
                               centroid_z=None,
                               #distance_from_prediction=None,
                               #prediction_matching_index = None,
                               n_vertices=0,
                               n_faces=0,
                               mesh=returned_file_path,
                               multiplicity=0,
                               sdf = None,
                               max_side_ratio = None,
                               bbox_volume_ratio = None,
                               max_hole_length=None,
                               run_time=run_time
                              )
            
            #raise Exception("to prevent writing because none were found")
            self.insert1(insert_dict,skip_duplicates=True)
            return
        
        #if there is one or more soma found, get the volume and side length checks
        max_side_ratio =  [np.max(side_length_ratios(m)) for m in total_soma_list]
        bbox_volume_ratio =  [soma_volume_ratio(m) for m in total_soma_list]
        dicts_to_insert = []


        for i,(current_soma,soma_sdf,sz_ratio,vol_ratio) in enumerate(zip(total_soma_list,total_soma_list_sdf,max_side_ratio,bbox_volume_ratio)):
            print("Trying to write off file")
            """ Currently don't need to export the meshes
            current_soma.export(f"{key['segment_id']}/{key['segment_id']}_soma_{i}.off")
            """
            auto_prediction_center = np.mean(current_soma.vertices,axis=0) / np.array([4,4,40])
            auto_prediction_center = auto_prediction_center.astype("int")
            print(f"Predicted Coordinates are {auto_prediction_center}")
            max_hole_length = tu.largest_hole_length(current_soma)
            
            returned_file_path = tu.write_h5_file(
                                            vertices=current_soma.vertices,
                                              faces=current_soma.faces,
                                              segment_id=segment_id,
                                              filename = f'{segment_id}_{i}.h5',
                                                filepath=str(du.get_somas_path())
                                             )



            insert_dict = dict(key,
                              soma_index=i+1,
                              centroid_x=auto_prediction_center[0],
                               centroid_y=auto_prediction_center[1],
                               centroid_z=auto_prediction_center[2],
                               n_vertices = len(current_soma.vertices),
                               n_faces = len(current_soma.faces),
                               mesh=returned_file_path,
                               multiplicity=len(total_soma_list),
                               sdf = np.round(soma_sdf,3),
                               max_side_ratio = np.round(sz_ratio,3),
                               bbox_volume_ratio = np.round(vol_ratio,3),
                               max_hole_length = np.round(max_hole_length,3),
                               run_time=np.round(run_time,4)
                              )



            dicts_to_insert.append(insert_dict)
        self.insert(dicts_to_insert,skip_duplicates=True)
        
        


# In[ ]:


#(schema.jobs & "table_name='__baylor_segment_centroid'").delete()


# In[ ]:


import time
start_time = time.time()
time.sleep(random.randint(0, 900))
print('Populate Started')
BaylorSegmentCentroid.populate(reserve_jobs=True, suppress_errors=True, order='random')
print('Populate Done')

print(f"Total time for BaylorSegmentCentroid populate = {time.time() - start_time}")

