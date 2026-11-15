"""Replay the three imported Part 1 VGGT Full checkpoints with their original package."""
import csv,hashlib,json,os,shutil,subprocess,sys,time,errno
from pathlib import Path
from p4.common import write_json,sha256_file
REV="bde9244f692cc497355a307826ce1ca25f39cd99"
repo=Path.cwd();root=repo/"runs/part2_supplement_v1/part1_replay";root.mkdir(parents=True,exist_ok=True)
package=root/"original_source";package.mkdir(exist_ok=True)
paths=subprocess.check_output(["git","ls-tree","-r","--name-only",REV,"p4"],text=True).splitlines()
for name in paths:
    if not name.endswith(".py"):continue
    target=package/name;target.parent.mkdir(parents=True,exist_ok=True)
    content=subprocess.check_output(["git","show",REV+":"+name])
    if target.exists():assert target.read_bytes()==content
    else:target.write_bytes(content)
digest=hashlib.sha256()
for p in sorted((package/"p4").glob("*.py")):digest.update(p.name.encode());digest.update(p.read_bytes())
package_digest=digest.hexdigest()
out=repo/"results/part2_supplement_v1";state={"status":"running","training_commit":REV,"package_sha256":package_digest,"rows":[]}
write_json(out/"part1_checkpoint_replay.json",state)
load=lambda p:json.loads(Path(p).read_text())
audit=repo/"results/hpc_setup/part1_handoff_901c5ae/archive"
receipt=load(repo/"results/hpc_setup/part1_handoff_901c5ae/checkpoint_import_receipt.json")
assert receipt["status"]=="completed"
assert not any(v for k,v in os.environ.items() if k.lower().endswith("_proxy") and k.lower()!="no_proxy")
try:
    for scene in ["Re10k-1","Waymo-405841","Dl3dv-2"]:
        source=audit/"cases"/scene/"vggt";prov=load(source/"train/provenance.json")
        assert prov["package_source_sha256"]==package_digest
        dataset=repo/"data/part1_replay_901c5ae"/scene;dataset.mkdir(parents=True,exist_ok=True)
        for name in ["manifest.json","reference.npz"]:
            original=audit/"datasets"/scene/name;target=dataset/name
            if target.exists():assert sha256_file(target)==sha256_file(original)
            else:shutil.copyfile(original,target)
        manifest=load(dataset/"manifest.json")
        for f in manifest["frames"]:
            original=repo/"data/prepared"/scene/f["image"];target=dataset/f["image"]
            assert sha256_file(original)==f["sha256"]
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                try:os.link(original,target)
                except OSError as exc:
                    if exc.errno not in {errno.EPERM,errno.EACCES,errno.EXDEV}:raise
                    shutil.copyfile(original,target)
        assert sha256_file(dataset/"manifest.json")==prov["manifest_sha256"]
        checkpoint=repo/"runs/part1_import_901c5ae/checkpoints"/scene/"vggt/step_007000.pt"
        record=next(r for r in receipt["checkpoints"] if r["scene"]==scene and r["method"]=="vggt")
        assert sha256_file(checkpoint)==record["sha256"]
        dest=root/scene;dest.mkdir(exist_ok=True)
        commands=[["-m","p4.train","render","--dataset",str(dataset),"--scene",str(source/"init/scene.npz"),"--checkpoint",str(checkpoint),"--output",str(dest/"renders")],
                  ["-m","p4.evaluate","renders","--dataset",str(dataset),"--render-dir",str(dest/"renders"),"--output",str(dest/"metrics.json")]]
        if not (dest/"metrics.json").exists():
            for index,args in enumerate(commands):
                with (dest/f"stage_{index}.log").open("w") as log:
                    result=subprocess.run([sys.executable]+args,cwd=package,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(scene+" replay stage "+str(index)+" failed")
        current=load(dest/"metrics.json");original=load(source/"metrics/step_007000.json")
        render=load(dest/"renders/render.json")
        assert current["complete"] and current["protocol"]==original["protocol"]
        assert current["split_hash"]==original["split_hash"] and current["test_frames"]==original["test_frames"]
        assert render["checkpoint_sha256"]==record["sha256"] and render["step"]==7000
        assert [r["image"] for r in current["per_frame"]]==[r["image"] for r in original["per_frame"]]
        delta={k:current[k]-original[k] for k in ["psnr","ssim","lpips"]}
        assert abs(delta["psnr"])<0.02 and abs(delta["ssim"])<0.002 and abs(delta["lpips"])<0.002
        row={"scene":scene,"test_frames":current["test_frames"],"checkpoint_sha256":record["sha256"],
             "original_metrics":{k:original[k] for k in delta},"replayed_metrics":{k:current[k] for k in delta},
             "replay_minus_original":delta,"metrics_path":str(dest/"metrics.json"),"same_original_package_and_manifest":True}
        state["rows"].append(row);write_json(out/"part1_checkpoint_replay.json",state)
        print(json.dumps({"Part1_checkpoint_replay":scene,**delta}),flush=True)
    state["status"]="completed";write_json(out/"part1_checkpoint_replay.json",state)
except Exception as exc:
    state.update(status="failed",error=str(exc));write_json(out/"part1_checkpoint_replay.json",state);raise
