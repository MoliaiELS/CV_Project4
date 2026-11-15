import json
from pathlib import Path
import numpy as np
import PIL
import cv2
from p4.common import sha256_file, write_json
from p4.data import load_manifest
expected={r["scene"]:r for r in json.loads(Path("results/data_audit/preparation.json").read_text())["scenes"]}
rows=[]
for scene,counts in [("Waymo-405841",(199,174,25,20)),("Dl3dv-2",(306,267,39,11)),("Re10k-1",(279,244,35,10))]:
    root=Path("data/prepared")/scene
    small=Path("data/part2_sparse")/scene
    m,s=load_manifest(root),load_manifest(small)
    assert len(m["frames"])==counts[0]
    assert sum(f["split"]=="train" for f in m["frames"])==counts[1]
    assert sum(f["split"]=="test" for f in m["frames"])==counts[2]
    assert sum(f["split"]=="train" for f in s["frames"])==counts[3]
    assert [f for f in m["frames"] if f["split"]=="test"]==[f for f in s["frames"] if f["split"]=="test"]
    for f in m["frames"]:
        assert sha256_file(root/f["image"])==f["sha256"]
    with np.load(small/"reference.npz",allow_pickle=False) as r:
        ids=[i for i,n in enumerate(r["names"]) if n in {f["image"] for f in s["frames"] if f["split"]=="train"}]
        centers=r["c2w"][ids,:3,3]
        singular=np.linalg.svd(centers-centers.mean(0),compute_uv=False)
        assert singular[1]/singular[0]>1e-5
    rows.append({"scene":scene,"full_train":counts[1],"sparse_train":counts[3],"test":counts[2],
                 "dense_split_hash":m["split_hash"],"sparse_split_hash":s["split_hash"],
                 "matches_B_published_prepared_hash":m["split_hash"]==expected[scene]["split_hash"],
                 "sparse_reference_center_singular_values":singular.tolist(),
                 "sparse_original_ids":s["split_source"]["selected_original_frame_ids"],
                 "unchanged_test":True,"all_dense_image_hashes_verified":True})
out={"versions":{"numpy":np.__version__,"opencv":cv2.__version__,"Pillow":PIL.__version__},"scenes":rows}
write_json("results/hpc_setup/data_preparation.json",out)
print(json.dumps(out,indent=2))
