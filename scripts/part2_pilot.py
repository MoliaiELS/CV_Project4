"""Run one sparse Part 2 pilot without changing the 7k comparison schedule."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.common import sha256_file, write_json
from p4.data import load_manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",default="Re10k-1")
    p.add_argument("--output",default="runs/part2_pilot/Re10k-1/sparse_700")
    p.add_argument("--stop-after",type=int,default=700)
    a=p.parse_args()
    if not 650 <= a.stop_after < 7000:
        raise ValueError("Pilot must pass the first densification and remain below formal 7k")
    root=Path(a.output)
    if root.exists():
        raise FileExistsError("Choose a new pilot output to preserve earlier evidence")
    root.mkdir(parents=True)
    dataset=Path("data/part2_sparse")/a.scene
    manifest=load_manifest(dataset)
    cfg=json.loads(Path("configs/part1.json").read_text())
    cfg["training"]["save_steps"]=sorted(set(cfg["training"]["save_steps"]+[100,a.stop_after]))
    config=root/"frozen_config.json"
    write_json(config,cfg)
    state={"status":"running","pilot_only":True,"formal_budget":7000,"pilot_steps":a.stop_after,
           "scene":a.scene,"train_frames":sum(f["split"]=="train" for f in manifest["frames"]),
           "test_frames":sum(f["split"]=="test" for f in manifest["frames"]),
           "split_hash":manifest["split_hash"],"stages":[],"proxy_environment":{k:v for k,v in os.environ.items() if k.lower().endswith("_proxy")}}
    if any(v for k,v in state["proxy_environment"].items() if k.lower()!="no_proxy"):
        raise RuntimeError("Proxy environment unexpectedly configured")
    write_json(root/"status.json",state)
    def run(name,args):
        print("START "+name,flush=True)
        started=time.monotonic()
        with (root/(name+".log")).open("w") as log:
            result=subprocess.run([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT)
        entry={"name":name,"returncode":result.returncode,"seconds":time.monotonic()-started,"command":args}
        state["stages"].append(entry);write_json(root/"status.json",state)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {root/name}.log")
        print("DONE "+json.dumps(entry),flush=True)
    try:
        run("initialize",["-m","p4.initialize","--dataset",str(dataset),"--output",str(root/"init"),
            "--method","vggt","--repo","third_party/vggt","--weights","weights/vggt/model.pt",
            "--seed","42","--max-points","100000","--confidence-quantile","0.5"])
        base=["-m","p4.train","train","--dataset",str(dataset),"--scene",str(root/"init/scene.npz"),
              "--output",str(root/"train"),"--config",str(config)]
        run("train_100",base+["--stop-after","100"])
        run("resume",base+["--resume",str(root/"train/checkpoints/step_000100.pt"),
                           "--stop-after",str(a.stop_after)])
        checkpoint=root/"train/checkpoints"/f"step_{a.stop_after:06d}.pt"
        run("render",["-m","p4.train","render","--dataset",str(dataset),"--scene",str(root/"init/scene.npz"),
             "--checkpoint",str(checkpoint),"--output",str(root/"renders")])
        run("evaluate",["-m","p4.evaluate","renders","--dataset",str(dataset),"--render-dir",str(root/"renders"),
             "--output",str(root/"metrics.json")])
        metrics=json.loads((root/"metrics.json").read_text())
        if not metrics["complete"]:
            raise RuntimeError("Evaluation incomplete: LPIPS assets missing")
        rendered=json.loads((root/"renders/render.json").read_text())
        logs=[json.loads(line) for line in (root/"train/train.jsonl").read_text().splitlines() if line]
        if logs[-1]["step"] != a.stop_after:
            raise RuntimeError("Training stopped before the pilot target")
        state.update(status="completed",metrics=metrics,alignment=rendered,
                     last_training_record=logs[-1],checkpoint_sha256=sha256_file(checkpoint))
        write_json(root/"status.json",state)
        print("COMPLETE "+json.dumps({"scene":a.scene,"pilot_steps":a.stop_after,
              "psnr":metrics["psnr"],"ssim":metrics["ssim"],"lpips":metrics["lpips"],
              "ate_rmse":rendered.get("ate_rmse")}),flush=True)
    except Exception as e:
        state.update(status="failed",error=str(e));write_json(root/"status.json",state)
        raise

if __name__=="__main__":
    main()
