"""Validate remaining scene inputs against source calibration and frozen baseline."""
import json,re,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from p4.common import sha256_file,source_version,write_json
from p4.data import load_manifest
from p4.initialize import preprocessing_geometry

base=Path("runs/part2_formal/Re10k-1/full_sparse_7000_v1")
baseline=json.loads((base/"status.json").read_text())
assert baseline["status"]=="completed"
assert source_version()["package_source_sha256"]==baseline["source"]["package_source_sha256"]
raw=Path("/data/disk1/shared/aiaa3201/project4_3d")
sys.path.insert(0,str(Path("third_party/vggt").resolve()))
from vggt.utils.load_fn import load_and_preprocess_images_square
rows={}
for scene,expected in {"Waymo-405841":(174,20,25,960,640),"Dl3dv-2":(267,11,39,256,256)}.items():
    full=Path("data/prepared")/scene; sparse=Path("data/part2_sparse")/scene
    a,b=load_manifest(full),load_manifest(sparse)
    train=lambda m:[f for f in m["frames"] if f["split"]=="train"]
    test=lambda m:[f for f in m["frames"] if f["split"]=="test"]
    assert (len(train(a)),len(train(b)),len(test(a)))==expected[:3]
    assert test(a)==test(b)
    source_frames={f["id"]:f for f in a["frames"]}
    for f in a["frames"]:
        assert (f["width"],f["height"])==expected[3:]
        assert sha256_file(full/f["image"])==f["sha256"]
    for f in b["frames"]:
        assert f==source_frames[f["id"]]
        assert sha256_file(sparse/f["image"])==f["sha256"]
    if scene=="Dl3dv-2":
        cameras={v["image_name"]:v for v in json.loads((raw/"DL3DV-2/cameras.json").read_text())}
    pose_error=k_error=0.0
    with np.load(full/"reference.npz",allow_pickle=False) as ref:
        lookup={str(n):i for i,n in enumerate(ref["names"])}
        for f in a["frames"]:
            if scene=="Dl3dv-2":
                c=cameras[f["id"]]; pose=np.eye(4)
                pose[:3,:3]=Rotation.from_quat(c["cam_quat"]).as_matrix()
                pose[:3,3]=c["cam_trans"]
                K=np.array([[c["fx"]*256,0,c["cx"]*256],[0,c["fy"]*256,c["cy"]*256],[0,0,1]])
            else:
                stem=Path(f["id"]).stem;src=raw/"405841/FRONT"
                pose=np.loadtxt(src/"gt"/(stem+".txt")).reshape(4,4)
                vals={k:float(v) for k,v in re.findall(r"\b(fx|fy|cx|cy)\s*:\s*([-+0-9.eE]+)",(src/"calib"/(stem+".txt")).read_text())}
                with Image.open(src/"rgb"/f["id"]) as image:ow,oh=image.size
                K=np.array([[vals["fx"]*f["width"]/ow,0,vals["cx"]*f["width"]/ow],
                            [0,vals["fy"]*f["height"]/oh,vals["cy"]*f["height"]/oh],[0,0,1]])
            i=lookup[f["image"]]
            pose_error=max(pose_error,float(np.max(np.abs(pose-ref["c2w"][i]))))
            k_error=max(k_error,float(np.max(np.abs(K-ref["K"][i]))))
        ids=[lookup[f["image"]] for f in train(b)]
        centers=ref["c2w"][ids,:3,3];singular=np.linalg.svd(centers-centers.mean(0),compute_uv=False)
        assert singular[1]/singular[0]>1e-5
    assert pose_error<1e-12 and k_error<1e-10
    path=full/a["frames"][0]["image"]
    images,bounds=load_and_preprocess_images_square([str(path)],target_size=518)
    A,valid,shape=preprocessing_geometry([expected[3:]],"vggt")
    assert np.allclose(bounds.numpy()[:,:4],valid,atol=1e-4) and tuple(images.shape[-2:])==shape
    rows[scene]={"status":"passed","full_train":expected[0],"sparse_train":expected[1],"test":expected[2],
                 "width":expected[3],"height":expected[4],"all_image_hashes_verified":True,
                 "unchanged_test_set":True,"reference_pose_max_abs_error":pose_error,
                 "reference_K_max_abs_error":k_error,"official_vggt_preprocessing_matches":True,
                 "full_split_hash":a["split_hash"],"sparse_split_hash":b["split_hash"],
                 "reference_sha256":sha256_file(full/"reference.npz")}
out={"baseline":str(base),"package_source_sha256":source_version()["package_source_sha256"],
     "baseline_config_sha256":sha256_file(base/"frozen_config.json"),"scenes":rows}
write_json("results/hpc_setup/remaining_scenes_preflight.json",out)
print(json.dumps(out,indent=2))
