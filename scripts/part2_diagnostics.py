"""Add paired camera diagnostics and an opacity-coverage proxy to preserved Part 2."""
import csv,gc,json,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from p4.common import write_json,source_version,sha256_file
from p4.data import load_manifest,load_scene
from p4.geometry import align_training_cameras,transform_poses
from p4.evaluate import evaluation_cameras
from p4.train import _rasterize,training_fingerprint
load=lambda p:json.loads(Path(p).read_text())
out=Path("results/part2_supplement_v1");out.mkdir(exist_ok=True,parents=True)
run=Path("runs/part2_supplement_v1/coverage");run.mkdir(exist_ok=True,parents=True)
baseline=load("runs/part2_formal/Re10k-1/full_sparse_7000_v1/status.json")
assert source_version()["package_source_sha256"]==baseline["source"]["package_source_sha256"]
assert not any(v for k,v in os.environ.items() if k.lower().endswith("_proxy") and k.lower()!="no_proxy")
rows=[];camera_rows=[];per_frame=[];reuse=[]
for scene in ["Re10k-1","Waymo-405841","Dl3dv-2"]:
    formal=Path("runs/part2_formal")/scene/"full_sparse_7000_v1"
    sparse=load_manifest(Path("data/part2_sparse")/scene)
    common=[f["image"] for f in sparse["frames"] if f["split"]=="train"]
    with np.load(Path("data/prepared")/scene/"reference.npz",allow_pickle=False) as ref:
        ref_names=ref["names"].copy();ref_c2w=ref["c2w"].copy();ref_K=ref["K"].copy()
    lookup={str(n):i for i,n in enumerate(ref_names)}
    reference=ref_c2w[[lookup[n] for n in common]]
    trajectories={"names":np.array(common),"reference":reference[:,:3,3]}
    fmetrics=load(formal/"full/metrics.json");smetrics=load(formal/"sparse/metrics.json")
    nt=fmetrics["test_frames"]
    gap=np.array([a["psnr"]-b["psnr"] for a,b in zip(fmetrics["per_frame"],smetrics["per_frame"])])
    selected={0,nt//2,nt-1,int(gap.argmax()),int(np.argmin([f["psnr"] for f in smetrics["per_frame"]]))}
    for kind in ["full","sparse"]:
        case=formal/kind;dataset=Path("data/prepared" if kind=="full" else "data/part2_sparse")/scene
        m=load_manifest(dataset);sc=load_scene(case/"init/scene.npz")
        paired=align_training_cameras(sc["names"],sc["c2w"],ref_names,ref_c2w,set(common))
        sc_index={str(n):i for i,n in enumerate(sc["names"])}
        est=sc["c2w"][[sc_index[n] for n in common]]
        aligned=transform_poses(est,paired["estimate_to_reference"])
        residual=np.linalg.norm(aligned[:,:3,3]-reference[:,:3,3],axis=1)
        rot=Rotation.from_matrix(np.swapaxes(reference[:,:3,:3],-1,-2)@aligned[:,:3,:3]).magnitude()*180/np.pi
        K=sc["K"][[sc_index[n] for n in common]];rk=ref_K[[lookup[n] for n in common]]
        trajectories[kind]=aligned[:,:3,3];trajectories[kind+"_error"]=residual
        center=reference[:,:3,3];singular=np.linalg.svd(center-center.mean(0),compute_uv=False)
        reference_path_length=float(np.linalg.norm(np.diff(center,axis=0),axis=1).sum())
        row={"scene":scene,"condition":kind,"common_frames":len(common),"paired_ate_rmse":paired["ate_rmse"],
             "all_available_train_ate":load(case/"renders/render.json")["ate_rmse"],
             "paired_rotation_mean_degrees":float(rot.mean()),"paired_rotation_max_degrees":float(rot.max()),
             "paired_focal_ratio_fx":float(np.median(K[:,0,0]/rk[:,0,0])),
             "paired_focal_ratio_fy":float(np.median(K[:,1,1]/rk[:,1,1])),
             "trajectory_singular_ratio_2_to_1":float(singular[1]/singular[0]),
             "reference_path_length":reference_path_length,
             "paired_ate_over_path_length":paired["ate_rmse"]/reference_path_length}
        camera_rows.append(row)
        checkpoint=case/"train/checkpoints/step_007000.pt"
        saved=torch.load(checkpoint,map_location="cuda",weights_only=False)
        assert saved["step"]==7000 and saved["fingerprint"]==training_fingerprint(dataset,case/"init/scene.npz",m)
        cameras=evaluation_cameras(dataset,case/"init/scene.npz")
        poses=cameras["c2w"].copy()
        poses[:,:3,3]=(poses[:,:3,3]-np.array(saved["normalization"]["origin"]))/saved["normalization"]["radius"]
        alpha_dir=run/scene/kind;alpha_dir.mkdir(parents=True,exist_ok=True)
        samples=[];replay_max=0
        with torch.inference_mode():
            for i,(name,pose,K) in enumerate(zip(cameras["names"],poses,cameras["K"])):
                with Image.open(dataset/str(name)) as im:width,height=im.size
                rgb,alpha,_=_rasterize(saved["splats"],torch.as_tensor(pose,dtype=torch.float32,device="cuda"),
                                      torch.as_tensor(K,dtype=torch.float32,device="cuda"),width,height,saved["sh_degree_active"])
                values=alpha[0,...,0].cpu().numpy()
                assert np.isfinite(values).all()
                row={"scene":scene,"condition":kind,"image":str(name),"mean_alpha":float(values.mean()),
                     "coverage_alpha_0_1":float((values>=0.1).mean()),"coverage_alpha_0_5":float((values>=0.5).mean()),
                     "coverage_alpha_0_9":float((values>=0.9).mean())}
                pixels=(rgb[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                with Image.open(case/"renders"/str(name)) as old:original=np.asarray(old.convert("RGB"))
                row["replay_max_pixel_difference"]=int(np.abs(pixels.astype(int)-original.astype(int)).max())
                replay_max=max(replay_max,row["replay_max_pixel_difference"])
                if i in selected:
                    target=alpha_dir/str(name);target.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(np.clip(values*255,0,255).round().astype(np.uint8)).save(target)
                    row["alpha_preview"]=str(target.resolve())
                per_frame.append(row);samples.append(row)
        rows.append({"scene":scene,"condition":kind,"test_frames":len(samples),
                     **{k:float(np.mean([s[k] for s in samples])) for k in ["mean_alpha","coverage_alpha_0_1","coverage_alpha_0_5","coverage_alpha_0_9"]},
                     "replay_max_pixel_difference":replay_max,"checkpoint_sha256":sha256_file(checkpoint)})
        del saved;gc.collect();torch.cuda.empty_cache()
        print(json.dumps({"coverage_complete":scene+"/"+kind,"coverage_alpha_0_5":rows[-1]["coverage_alpha_0_5"],"replay_max":replay_max}),flush=True)
    np.savez_compressed(out/(scene+"_paired_trajectories.npz"),**trajectories)
    bcase=Path("results/hpc_setup/part1_handoff_901c5ae/archive/cases")/scene/"vggt"
    bsc=load_scene(bcase/"init/scene.npz")
    alignment=align_training_cameras(bsc["names"],bsc["c2w"],ref_names,ref_c2w,set(common))
    bm=load(bcase/"metrics/step_007000.json")
    reuse.append({"scene":scene,"A_full_paired_ate":next(r["paired_ate_rmse"] for r in camera_rows if r["scene"]==scene and r["condition"]=="full"),
                  "B_full_paired_ate":alignment["ate_rmse"],"B_original_all_train_ate":load(bcase/"renders/step_007000/render.json")["ate_rmse"],
                  **{k+"_B_minus_A":bm[k]-fmetrics[k] for k in ["psnr","ssim","lpips"]},
                  "same_training_config":load(bcase/"train/config.json")==load(formal/"full/train/config.json"),
                  "B_checkpoint_copy":str((Path("runs/part1_import_901c5ae/checkpoints")/scene/"vggt/step_007000.pt").resolve())})
for name,items in [("coverage",rows),("camera_diagnostics",camera_rows),("part1_reuse_comparison",reuse)]:
    with (out/(name+".csv")).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
write_json(out/"coverage_per_frame.json",per_frame)
write_json(out/"diagnostics.json",{"status":"completed","coverage":rows,"camera_diagnostics":camera_rows,"part1_reuse":reuse,
           "coverage_definition":"Fraction of held-out rendered pixels whose accumulated Gaussian alpha exceeds the threshold; a visibility/support proxy, not ground-truth 3D completeness or geometric accuracy.",
           "camera_protocol":"One Sim3 fitted on the same sparse TRAIN camera IDs per condition; reference-trajectory units. ATE is constant during fixed-camera GS training.",
           "selected_alpha_previews":"First/middle/last test, lowest Sparse PSNR, and largest Full-minus-Sparse PSNR gap.",
           "test_images_used_for_training":False,"source_unchanged":source_version()["package_source_sha256"]==baseline["source"]["package_source_sha256"]})
