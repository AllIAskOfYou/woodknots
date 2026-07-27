import pycolmap
from pathlib import Path

data = Path("data/matching")
images = data / "resized_512"
database = data / "database.db"
sparse = data / "sparse"
sparse.mkdir(exist_ok=True)

pycolmap.extract_features(database, images)
print("Extracted features")

# Sequential matching, no vocab-tree loop detection (avoids the FAISS/OpenBLAS crash)
pycolmap.match_sequential(database, pairing_options=pycolmap.SequentialPairingOptions(
    overlap=10,
    loop_detection=False,
))
print("Sequential matching done")

maps = pycolmap.incremental_mapping(database, images, sparse)
maps[0].write(sparse)   # writes cameras.bin, images.bin, points3D.bin
print("Incremental mapping completed")

# sanity check before spending time on dense/meshing
rec = pycolmap.Reconstruction(sparse)
print(rec.summary())

dense = data / "dense"

pycolmap.undistort_images(dense, sparse, images)
pycolmap.patch_match_stereo(dense)   # GPU/CUDA required here
print("Patch-match stereo completed")
pycolmap.stereo_fusion(
    dense / "fused.ply", dense,
)
print("Dense reconstruction completed")

# meshing
pycolmap.poisson_meshing(dense / "fused.ply", dense / "meshed-poisson.ply")
print("Meshing completed")