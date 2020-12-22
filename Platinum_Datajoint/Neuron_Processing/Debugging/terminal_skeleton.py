if __name__ == "__main__":
    mesh_file = "./mesh.off"
    sk_filename = "terminal_sk_75"


    from os import sys
    sys.path.append("/meshAfterParty/")

    import time
    import skeleton_utils as sk
    import system_utils as su

    st_time = time.time()
    terminal_sk = sk.skeleton_cgal(mesh_path=mesh_file,      quality_speed_tradeoff=0.2,                  medially_centered_speed_tradeoff=0.2,
            area_variation_factor=0.0001,
                                 max_iterations=500,
                                   min_edge_length=75,
                                  )
    print(f"\n\n Total time for skeletonization = {time.time() - st_time}")
    su.compressed_pickle(terminal_sk,sk_filename)