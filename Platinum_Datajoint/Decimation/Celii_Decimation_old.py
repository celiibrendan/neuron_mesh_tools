#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To create a table with the segment id's that will
be used to pick which meshes to decimate based on the nuclei table

"""


# In[ ]:


from os import sys
sys.path.append("/meshAfterParty/")
from importlib import reload

import datajoint as dj
from pathlib import Path

import datajoint_utils as du
du = reload(du)

import time


# In[ ]:


import minfig
du.config_celii()
du.set_minnie65_config_segmentation(minfig)
du.print_minnie65_config_paths(minfig)

#configuring will include the adapters
minnie,schema = du.configure_minnie_vm()


# # Christos Setup

# In[ ]:


import datajoint as dj
import numpy as np
import h5py
import os

from collections import namedtuple

import argparse
import sys
import os
import subprocess
import traceback

import datajoint as dj
import warnings
import json
import os
from pathlib import Path


# In[ ]:


from meshlab import Decimator
temporary_folder = 'decimation_temp'
meshlab_scripts = {}


# In[ ]:


from minfig.adapters import *


# In[ ]:


from minfig.minnie65_config import external_decimated_mesh_path
external_decimated_mesh_path

@minnie.schema
class Decimation(dj.Computed):
#     definition = minnie.Decimation.describe(printout=False)
    key_source = minnie.Mesh.proj() * (minnie.DecimationConfig & 'decimation_ratio=0.25')

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

        if decimation_ratio not in meshlab_scripts:
            meshlab_scripts[decimation_ratio] = Decimator(decimation_ratio, temporary_folder, overwrite=False)
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


# # Experimenting with the relation

# In[ ]:


import random
import time

segment_rel = minnie.Mesh() & minnie.SegToDecimateFromNuclei()

# Random sleep delay to avoid concurrent key_source queries from hangin
time.sleep(random.randint(0, 900))
print('Populate Started')
Decimation.populate(segment_rel, reserve_jobs=True, suppress_errors=True, order='random')
print('Populate Done')


# In[ ]:




