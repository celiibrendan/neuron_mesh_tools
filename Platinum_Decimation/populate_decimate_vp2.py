#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datajoint as dj
from meshlab import Decimator
from minfig import * # Required for the adapters to be used with locally defined tables

# Virtual module accessors
minnie = configure_minnie(return_virtual_module=True) # virtual module with the adapted attribute for mesh access from .h5 files
schema = dj.schema("microns_minnie65_01")


# In[2]:


def concatenated_rel(cls, core_segment=None, version=-1, return_with_meshes=False):
    """
    Returns all. You can restrict by a core_segment first though.

    :param core_segment: The core segment(s) to restrict by. If left empty will fetch all.
    :param version: The default of -1 will fetch the highest version for each core segment
        and its subsegments. If you happen to explicitely pass False, it will ignore version.
    :param return_with_meshes: When set to true or 'Decimation' will default
        to using the Decimation table for the meshes, otherwise 'Mesh' will
        choose the Mesh table with the original meshes.
    """

    subsegment_rel = cls.Subsegment.proj()

    if core_segment is not None:
        try:
            subsegment_rel &= [dict(segment_id=segment_id) for segment_id in core_segment]
        except TypeError:
            subsegment_rel &= dict(segment_id=core_segment)

    if version == -1:
        version_rel = dj.U('segment_id').aggr(subsegment_rel, version='max(version)')
    elif version is False:
        version_rel = subsegment_rel
    else:
        version_rel = subsegment_rel & dict(version=version)

    a_rel = dj.U('segment_id') & version_rel
    b_rel = dj.U('segment_id') & (subsegment_rel & version_rel).proj(_='segment_id', segment_id='subsegment_id')
    c_rel = a_rel + b_rel

    if return_with_meshes:
        if isinstance(return_with_meshes, str) and return_with_meshes.lower() == 'mesh':
            c_rel = minnie.Mesh & c_rel
        else:
            c_rel = minnie.Decimation & c_rel

    return c_rel


# In[3]:


temporary_folder = 'decimation_temp'
meshlab_scripts = {}


# In[5]:


@schema
class DecimationTest(dj.Computed):
    definition="""
    -> minnie.Mesh
    -> minnie.DecimationConfig
    ---
    n_vertices           : bigint                       
    n_faces              : bigint                       
    mesh                 : <decimated_mesh>             # in-place path to the hdf5 (decimated) mesh file
    INDEX (mesh)
    """
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



    key_source = minnie.Mesh.proj() * minnie.DecimationConfig

    def make(self, key):
        print(key)
        mesh = (minnie.Mesh & key).fetch1('mesh')
        segment_id = key['segment_id']
        version = key['version']
        decimation_ratio = key['decimation_ratio']

        # Here is where creates the decimator
        if decimation_ratio not in meshlab_scripts:
            meshlab_scripts[decimation_ratio] = Decimator(decimation_ratio, temporary_folder, overwrite=False)
            print("making new meshlab_scrip")
        mls_func = meshlab_scripts[decimation_ratio]

        try:
            expected_filepath = os.path.join(external_decimated_mesh_path, f'{segment_id}_{version}_{int(decimation_ratio*100):02}.h5')
            #if not os.path.isfile(expected_filepath):
            if True:
                #running the actual decimation
                new_mesh = mls_func(mesh.vertices, mesh.faces, segment_id,random_port=True)
                #new_vertices, new_faces = new_mesh.vertices, new_mesh.faces
                #makes the entry into the table
#                 self.make_entry(
#                     segment_id=segment_id,
#                     version=version,
#                     decimation_ratio=decimation_ratio,
#                     vertices=new_vertices,
#                     faces=new_faces
#                 )
            else:
                print('File already exists.')
                raise FileExistsError('{} already exists'.format(expected_filepath))
#                 with h5py.File(expected_filepath, 'r') as hf:
#                     vertices = hf['vertices'][()].astype(np.float64)
#                     faces = hf['faces'][()].reshape(-1, 3).astype(np.uint32)
#                 self.insert1(dict(key, n_vertices=len(vertices), n_faces=len(faces), mesh=expected_filepath))
        #where throws the exception
        except Exception as e:
            minnie.DecimationError.insert1(dict(key, log=str(e)))
            print(e)
            raise e


# In[ ]:



import random
import time

allen_segment_rel = minnie.AllenSegmentCentroid
prioritize_allen_soma = allen_segment_rel * (minnie.DecimationConfig & {'version': 0, 'decimation_ratio': 0.25}) & 'segment_id!=0'

#     segment_rel = concatenated_rel(minnie.FromNeuromancer())

# Random sleep delay to avoid concurrent key_source queries from hangin
time.sleep(random.randint(0, 180))# * 15))
print('Populate Started')
DecimationTest.populate(prioritize_allen_soma, reserve_jobs=True, suppress_errors=True, order='random')
#     Decimation.populate(segment_rel, reserve_jobs=True, suppress_errors=True, order='random')
#     Decimation.populate({'decimation_ratio': 0.25}, reserve_jobs=True, suppress_errors=True, order='random')
print('Populate Done')


# In[ ]:


import random
random.randint(100,10000)


# In[ ]:




