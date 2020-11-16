#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
**MORE ROBUST SOMA FINDNG ALGORITHM**
Purpose: Creates the table that will store the soma centroid from meshes used for testing

Pseudocode: 
1) Pull down the mesh
2) Create a folder where all of the mesh pieces will be stored
3) Run the soma extraction to get the soma meshes
4) Calculate the soma centers and save the distance from the predicted soma centers
5) Things then to save off:
- index of the soma (based on tehe number of somas found for that one mesh)
- the soma center
- Distance from predicted soma center
- Mesh of the soma
- Time it took
6) Delete all the temporary files that were used for that certain mesh
"""


# In[2]:


# Testing 
    import datajoint as dj
    import numpy as np
    m65 = dj.create_virtual_module('m65', 'microns_minnie65_01')
    schema = dj.schema("microns_minnie65_01")
    dj.config["display.limit"] = 30

    import minfig
    minnie = minfig.configure_minnie(return_virtual_module=True)


# In[3]:


import cgal_Segmentation_Module as csm
from whole_neuron_classifier_datajoint_adapted import extract_branches_whole_neuron
import whole_neuron_classifier_datajoint_adapted as wcda 
import time
import trimesh
import numpy as np
import datajoint as dj
import os
from meshlab import Decimator , Poisson
from pathlib import Path
from pykdtree.kdtree import KDTree
from soma_extraction_utils import *


# In[4]:


decimation_version = 0
decimation_ratio = 0.25
import time


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
    soma_vertices=NULL        : longblob                # array of vertices
    soma_faces=NULL           : longblob                   # array of faces
    multiplicity=NULL         : tinyint unsigned             # the number of somas found for this base segment
    sdf=NULL                  : double                       # sdf width value for the soma
    max_side_ratio=NULL       : double                       # the maximum of the side length ratios used for check if soma
    bbox_volume_ratio=NULL    : double                       # ratio of bbox (axis aligned) volume to mesh volume to use for check if soma
    run_time=NULL : double                   # the amount of time to run (seconds)

    """
    
    #key_source = (minnie.Decimation & (dj.U("segment_id") & (m65.AllenSegmentCentroid & "status=1").proj()) & "version=" + str(decimation_version))
    #key_source = minnie.Decimation() & (m65.AllenMultiSomas() & "status > 0").proj()
    segments = (m65.Mesh() & f'n_vertices  > {np.str(np.round(np.exp(12)).astype(np.int))}' & f'n_vertices  < {np.str(np.round(np.exp(15)).astype(np.int))}')
    key_source =  minnie.Decimation.proj(decimation_version='version') & segments.proj() & "decimation_version=" + str(decimation_version)
    
    def make(self,key):
        #get the mesh data
        print(f"\n\n\n---- Working on {key['segment_id']} ----")
        
        new_mesh = (minnie.Decimation() & key).fetch1("mesh")
        current_mesh_verts,current_mesh_faces = new_mesh.vertices,new_mesh.faces
        
        segment_id = key["segment_id"]
        
        (total_soma_list, 
         run_time, 
         total_soma_list_sdf) = extract_soma_center(
                            segment_id,
                            current_mesh_verts,
                            current_mesh_faces,
                            outer_decimation_ratio= 0.25,
                            large_mesh_threshold = 60000,
                            large_mesh_threshold_inner = 40000,
                            soma_width_threshold = 0.32,
                            soma_size_threshold = 20000,
                           inner_decimation_ratio = 0.25,
                           volume_mulitplier=7,
                           side_length_ratio_threshold=3,
                            soma_size_threshold_max=192000,
                            delete_files=True
        )
        
        print(f"Run time was {run_time} \n    total_soma_list = {total_soma_list}"
             f"\n    with sdf values = {total_soma_list_sdf}")
        
        #check if soma list is empty and did not find soma
        if len(total_soma_list) <= 0:
            print("There were no somas found for this mesh so just writing empty data")
            insert_dict = dict(key,
                              soma_index=0,
                              centroid_x=None,
                               centroid_y=None,
                               centroid_z=None,
                               #distance_from_prediction=None,
                               #prediction_matching_index = None,
                               n_vertices=0,
                               n_faces=0,
                               soma_vertices=None,
                               soma_faces=None,
                               multiplicity=0,
                               sdf = None,
                               max_side_ratio = None,
                               bbox_volume_ratio = None,
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



            insert_dict = dict(key,
                              soma_index=i+1,
                              centroid_x=auto_prediction_center[0],
                               centroid_y=auto_prediction_center[1],
                               centroid_z=auto_prediction_center[2],
        #                                distance_from_prediction=error_distance,
        #                                prediction_matching_index = prediction_matching_index,
                               n_vertices = len(current_soma.vertices),
                               n_faces = len(current_soma.faces),
                               soma_vertices=current_soma.vertices,
                               soma_faces=current_soma.faces,
                               multiplicity=len(total_soma_list),
                               sdf = np.round(soma_sdf,3),
                               max_side_ratio = np.round(sz_ratio,3),
                               bbox_volume_ratio = np.round(vol_ratio,3),
                               run_time=np.round(run_time,4)
                              )



            dicts_to_insert.append(insert_dict)

        self.insert(dicts_to_insert,skip_duplicates=True)


# In[5]:


#(SomaCentroidValidation & "segment_id=90231147459666747").delete()
#(schema.jobs & "table_name='__baylor_segment_centroid'").delete()
#m65.SomaCentroidValidation()


# In[6]:


"""
Actually will do the population
"""

#(schema.jobs & "table_name='__whole_auto_annotations_label_clusters3'")#.delete()
dj.config["enable_python_native_blobs"] = True

import time
start_time = time.time()
BaylorSegmentCentroid.populate(reserve_jobs=True)
print(f"Total time for BaylorSegmentCentroid populate = {time.time() - start_time}")

