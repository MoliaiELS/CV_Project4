"""Evaluate saved Part 2 checkpoints at the five frozen iteration points."""
import csv,json,os,subprocess,sys,time
from pathlib import Path
from p4.common import write_json,sha256_file,source_version
SCENES=["Re10k-1","Waymo-405841","Dl3dv-2"];STEPS=[500,1000,2000,4000,7000]
root=Path("runs/part2_supplement_v1");root.mkdir(parents=True,exist_ok=True)
out=Path("results/part2_supplement_v1");out.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(Path(p).read_text())
baseline=load("runs/part2_formal/Re10k-1/full_sparse_7000_v1/status.json")
digest=baseline["source"]["package_source_sha256"]
assert source_version()["package_source_sha256"]==digest
assert not any(v for k,v in os.environ.items() if k.lower().endswith("_proxy") and k.lower()!="no_proxy")
state={"status":"running","source_sha256":digest,"steps":STEPS,"points":[],"stages":[],"proxy_environment":{}}
write_json(out/"convergence_status.json",state)
def run(label,args):
    assert source_version()["package_source_sha256"]==digest
    start=time.monotonic();log=root/(label+".log")
    with log.open("w") as f:r=subprocess.run([sys.executable]+args,stdout=f,stderr=subprocess.STDOUT)
    state["stages"].append({"name":label,"returncode":r.returncode,"seconds":time.monotonic()-start,"command":args})
    write_json(out/"convergence_status.json",state)
    if r.returncode:raise RuntimeError(label+" failed: "+str(log))
try:
    for scene in SCENES:
        formal=Path("runs/part2_formal")/scene/"full_sparse_7000_v1"
        assert load(formal/"status.json")["status"]=="completed"
        for kind in ["full","sparse"]:
            case=formal/kind;dataset=Path("data/prepared" if kind=="full" else "data/part2_sparse")/scene
            final=load(case/"metrics.json")
            training={x["step"]:x for x in map(json.loads,(case/"train/train.jsonl").read_text().splitlines())}
            for step in STEPS:
                checkpoint=case/f"train/checkpoints/step_{step:06d}.pt"
                dest=root/"convergence"/scene/kind/f"step_{step:06d}"
                if step==7000:
                    metrics=final;render=load(case/"renders/render.json");metrics_path=case/"metrics.json"
                else:
                    metrics_path=dest/"metrics.json";dest.mkdir(parents=True,exist_ok=True)
                    if not metrics_path.exists():
                        label=f"{scene}_{kind}_{step:06d}"
                        run(label+"_render",["-m","p4.train","render","--dataset",str(dataset),"--scene",str(case/"init/scene.npz"),
                                            "--checkpoint",str(checkpoint),"--output",str(dest/"renders")])
                        run(label+"_evaluate",["-m","p4.evaluate","renders","--dataset",str(dataset),"--render-dir",str(dest/"renders"),"--output",str(metrics_path)])
                    metrics=load(metrics_path);render=load(dest/"renders/render.json")
                assert metrics["complete"] and metrics["protocol"]==final["protocol"]
                assert metrics["split_hash"]==final["split_hash"] and metrics["test_frames"]==final["test_frames"]
                assert [x["image"] for x in metrics["per_frame"]]==[x["image"] for x in final["per_frame"]]
                assert render["step"]==step and render["checkpoint_sha256"]==sha256_file(checkpoint)
                row={"scene":scene,"condition":kind,"step":step,**{k:metrics[k] for k in ["psnr","ssim","lpips","test_frames"]},
                     "training_seconds":training[step]["training_seconds"],"num_gaussians":training[step]["num_gaussians"],
                     "checkpoint_sha256":render["checkpoint_sha256"],"metrics_path":str(metrics_path.resolve()),
                     "source":"preserved formal final" if step==7000 else "new evaluation of preserved checkpoint"}
                state["points"].append(row);write_json(out/"convergence_status.json",state)
                print(json.dumps({"completed_points":len(state["points"]),"total":30,**{k:row[k] for k in ["scene","condition","step","psnr"]}}),flush=True)
    assert len(state["points"])==30
    with (out/"convergence.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(state["points"][0]));writer.writeheader();writer.writerows(state["points"])
    state.update(status="completed",new_evaluation_points=24,preserved_final_points=6,
                 total_per_frame_evaluations=sum(x["test_frames"] for x in state["points"]))
    write_json(out/"convergence_status.json",state)
except Exception as e:
    state.update(status="failed",error=str(e));write_json(out/"convergence_status.json",state);raise
