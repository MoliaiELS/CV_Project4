"""Audit camera interpretation without modifying reconstruction or held-out poses."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from p4.common import write_json
from p4.data import load_manifest, load_scene
from p4.geometry import project_points, transform_poses
from p4.initialize import preprocessing_geometry
from p4.train import normalized_geometry

root = Path("data/prepared/Re10k-1")
run = Path("runs/part2_pilot/Re10k-1/sparse_700")
manifest = load_manifest(root)
raw = Path("/data/disk1/shared/aiaa3201/project4_3d/Re10k-1")
source = {x["image_name"]:x for x in json.loads((raw/"cameras.json").read_text())}
with np.load(root/"reference.npz",allow_pickle=False) as ref:
    refs = {str(n):(p.copy(),k.copy()) for n,p,k in zip(ref["names"],ref["c2w"],ref["K"])}
ref_errors = {"rotation":0.0,"translation":0.0,"intrinsics":0.0}
for f in manifest["frames"]:
    rec = source[f["id"]]; pose, K = refs[f["image"]]
    independent_R = Rotation.from_quat(rec["cam_quat"]).as_matrix()
    independent_K = np.array([[rec["fx"]*f["width"],0,rec["cx"]*f["width"]],
                              [0,rec["fy"]*f["height"],rec["cy"]*f["height"]],[0,0,1]])
    for key,err in [("rotation",np.max(np.abs(pose[:3,:3]-independent_R))),
                    ("translation",np.max(np.abs(pose[:3,3]-rec["cam_trans"]))),
                    ("intrinsics",np.max(np.abs(K-independent_K)))]:
        ref_errors[key] = max(ref_errors[key],float(err))
assert max(ref_errors.values()) < 1e-12

scene = load_scene(run/"init/scene.npz")
paths = [root / str(n) for n in scene["names"]]
sizes = [Image.open(p).size for p in paths]
sys.path.insert(0,str(Path("third_party/vggt").resolve()))
from vggt.utils.load_fn import load_and_preprocess_images_square
official_images, official_bounds = load_and_preprocess_images_square([str(p) for p in paths],target_size=518)
A,bounds,shape = preprocessing_geometry(sizes,"vggt")
assert np.allclose(bounds,official_bounds.numpy()[:,:4]) and tuple(official_images.shape[-2:])==shape
stored = json.loads((run/"init/scene.json").read_text())
assert np.array_equal(A,np.array(stored["canonical_to_model"]))

projection_errors, normalization_errors = [], []
np_poses, np_points, norm = normalized_geometry(scene)
for i,(pose,K) in enumerate(zip(scene["c2w"],scene["K"])):
    points = scene["points"][::97].astype(np.float64)
    uv,z = project_points(points,pose,K)
    w2c = np.linalg.inv(pose)
    independent = cv2.projectPoints(points,cv2.Rodrigues(w2c[:3,:3])[0],w2c[:3,3],K,None)[0][:,0]
    good=(z>0.01)&(uv[:,0]>=0)&(uv[:,0]<256)&(uv[:,1]>=0)&(uv[:,1]<256)
    assert good.any()
    projection_errors.append(float(np.max(np.abs(uv[good]-independent[good]))))
    mapped,_ = project_points(np_points[::97],np_poses[i],K)
    normalization_errors.append(float(np.max(np.abs(uv[good]-mapped[good]))))
print("PROJECTION_DIAGNOSTIC", max(projection_errors), max(normalization_errors), flush=True)
# Model rotations are stored as float32; Rodrigues projects them back onto SO(3).
assert max(projection_errors)<1e-3
assert max(normalization_errors)<0.01

def fundamental(c1,k1,c2,k2):
    relative=np.linalg.inv(c2)@c1
    tx,ty,tz=relative[:3,3]
    skew=np.array([[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]])
    return np.linalg.inv(k2).T@skew@relative[:3,:3]@np.linalg.inv(k1)
def sampson(F,p,q):
    p=np.column_stack([p,np.ones(len(p))]); q=np.column_stack([q,np.ones(len(q))])
    fp=p@F.T; fq=q@F
    return np.abs(np.sum(q*fp,axis=1))/np.sqrt(np.sum(fp[:,:2]**2+fq[:,:2]**2,axis=1)).clip(1e-15)

sift=cv2.SIFT_create(nfeatures=2000)
features=[]
for path in paths:
    gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    key,des=sift.detectAndCompute(gray,None); features.append((key,des))
pairs=[]
cv2.setRNGSeed(42)
for i in range(len(paths)-1):
    j=i+1; k1,d1=features[i]; k2,d2=features[j]
    matches=[m for m,n in cv2.BFMatcher().knnMatch(d1,d2,k=2) if m.distance<0.7*n.distance]
    if len(matches)<12: continue
    p=np.float32([k1[m.queryIdx].pt for m in matches]); q=np.float32([k2[m.trainIdx].pt for m in matches])
    independent_F,mask=cv2.findFundamentalMat(p,q,cv2.USAC_MAGSAC,1.0,0.999,10000)
    if independent_F is None or mask is None: continue
    keep=mask.ravel().astype(bool); p,q=p[keep],q[keep]
    rp1,rk1=refs[str(scene["names"][i])]; rp2,rk2=refs[str(scene["names"][j])]
    Fs={"reference":fundamental(rp1,rk1,rp2,rk2),
        "vggt":fundamental(scene["c2w"][i],scene["K"][i],scene["c2w"][j],scene["K"][j]),
        "train_image_ransac":independent_F}
    pairs.append({"a":str(scene["names"][i]),"b":str(scene["names"][j]),"inliers":len(p),
                  "median_sampson_px":{k:float(np.median(sampson(F,p,q))) for k,F in Fs.items()}})
report={"status":"passed","reference_frames_checked":len(manifest["frames"]),
    "reference_interpretation":"xyzw quaternion, OpenCV c2w, normalized intrinsics multiply width/height",
    "source_reference_max_abs_error":ref_errors,"official_vggt_padding_resize_matches":True,
    "opencv_projection_max_abs_error_px":max(projection_errors),
    "training_normalization_projection_max_abs_error_px":max(normalization_errors),
    "train_only_feature_pairs":pairs,
    "conclusion":"No coordinate, intrinsic resize, frame-order or projection implementation error found. VGGT camera errors are retained as model-estimation error; no reference cameras are substituted into initialization/training.",
    "parameters_or_test_alignment_changed":False}
write_json("results/hpc_setup/camera_audit.json",report)
print(json.dumps(report,indent=2))
