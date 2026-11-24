# External code and weights

Model inference comes from the fixed ModelTC/LightX2V inference PR, and the original model project/package pinned in configs/nvidia/source-lock.json. No external model implementation is vendored in this evaluation PR. Check each source’s license and model/data access terms before downloading or redistribution.



Mesh loading/sampling and nearest-neighbor search use trimesh and SciPy. The point-cloud metric definitions and GSO first-thumbnail selection belong to the project evaluation protocol. SF3D preprocessing/model code comes from the pinned inference or native source, not a second local implementation. GSO, DINOv2, LAION CLIP and rembg weights remain external.
