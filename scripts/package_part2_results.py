"""Package small, verified Part 2 result artifacts on the HPC."""
import json,os,subprocess,tarfile,zipfile
from pathlib import Path
from p4.common import sha256_file,source_version,write_json
repo=Path.cwd()
out=repo/"results/part2_formal_7000"
summary=json.loads((out/"comparison.json").read_text())
assert len(summary["rows"])==6 and all(v["completed"] for v in summary["verification"].values())
assert not any(v for k,v in os.environ.items() if k.lower().endswith("_proxy") and k.lower()!="no_proxy")
src=[]
for directory in ["p4","scripts","tests","configs","docs"]:
    src.extend(p for p in (repo/directory).rglob("*") if p.is_file() and p.suffix in [".py",".json",".md"] and "__pycache__" not in p.parts)
for name in ["pyproject.toml","requirements.txt","README.md",".gitignore"]:
    if (repo/name).is_file():src.append(repo/name)
src=sorted(set(src))
with tarfile.open(out/"source_snapshot.tar.gz","w:gz") as archive:
    for path in src:archive.add(path,arcname=str(path.relative_to(repo)),recursive=False)
diff=subprocess.check_output(["git","diff","--binary"],cwd=repo)
(out/"tracked_changes.patch").write_bytes(diff)
write_json(out/"source_provenance.json",{
    "source":source_version(),"files":{str(p.relative_to(repo)):sha256_file(p) for p in src},
    "snapshot_sha256":sha256_file(out/"source_snapshot.tar.gz"),
    "note":"Snapshot includes modified and untracked source files; trained p4 digest equals the frozen Re10k baseline.",
    "no_proxy":True})
files=[p for p in out.iterdir() if p.is_file() and p.suffix in [".json",".csv",".md",".gz",".patch"] and p.name!="bundle_manifest.json"]
names=["status.json","frozen_config.json","comparison.json","verification_complete.json","camera_diagnostics_verified.json","report_verified.md","preview_verified.jpg"]
for scene in ["Re10k-1","Waymo-405841","Dl3dv-2"]:
    root=repo/"runs/part2_formal"/scene/"full_sparse_7000_v1"
    files.extend(root/name for name in names)
    for condition in ["full","sparse"]:
        case=root/condition
        files.extend(case/name for name in ["metrics.json","init/scene.json","train/config.json","train/provenance.json","train/train.jsonl","renders/render.json"])
for name in ["remaining_scenes_preflight.json","camera_audit.json","source_archive_identity.json","b_preprocessing_comparison.json","source_artifacts.json"]:
    files.append(repo/"results/hpc_setup"/name)
assert all(p.is_file() for p in files)
write_json(out/"bundle_manifest.json",{
    "files":{str(p.relative_to(repo)):sha256_file(p) for p in files},
    "checkpoint_locations":[{"scene":r["scene"],"condition":r["condition"],"path":r["run_path"]+"/"+r["condition"]+"/train/checkpoints/step_007000.pt","sha256":r["checkpoint_sha256"]} for r in summary["rows"]],
    "test_render_images":198,"checkpoints_included":False,"raw_test_renders_included":False,"all_artifacts_generated_on_HPC":True})
files.append(out/"bundle_manifest.json")
with zipfile.ZipFile(out/"summary_bundle.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for path in files:archive.write(path,arcname=str(path.relative_to(repo)))
with zipfile.ZipFile(out/"summary_bundle.zip") as archive:
    assert archive.testzip() is None
print(json.dumps({"bundle":str(out/"summary_bundle.zip"),"bytes":(out/"summary_bundle.zip").stat().st_size,
                  "sha256":sha256_file(out/"summary_bundle.zip"),"files":len(files)},indent=2))
