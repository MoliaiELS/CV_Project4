"""Copy and verify the shared Part 1 checkpoints without loading pickle payloads."""
import hashlib,json,os,shutil,time
from datetime import datetime,timezone
from pathlib import Path
from p4.common import sha256_file,source_version,write_json
repo=Path.cwd()
src=Path("/data/disk2/home/yguo704/CV_Project4-merged/output/handoff/part1-checkpoints-901c5ae")
dst=repo/"runs/part1_import_901c5ae/checkpoints"
audit_root=repo/"results/hpc_setup/part1_handoff_901c5ae"
audit=json.loads((audit_root/"audit.json").read_text())
manifest=json.loads((src/"manifest.json").read_text())
assert manifest["handoff_commit"]==audit["github_commit"]
assert manifest["training_commit"]=="bde9244f692cc497355a307826ce1ca25f39cd99" and manifest["step"]==7000
assert not any(v for k,v in os.environ.items() if k.lower().endswith("_proxy") and k.lower()!="no_proxy")
expected={c["scene"]+"/"+c["method"]+"/step_007000.pt":c for c in audit["large_checkpoints"]}
listed={c["file"]:c for c in manifest["checkpoints"]}
sums={}
for line in (src/"SHA256SUMS").read_text().splitlines():
    digest,name=line.split(maxsplit=1);sums[name.lstrip("*")]=digest
assert len(expected)==9 and expected.keys()==listed.keys()==sums.keys()
for name,old in expected.items():
    new=listed[name];assert new["sha256"]==sums[name]==old["sha256"] and new["bytes"]==old["bytes"]
    path=src/name;assert path.resolve().is_relative_to(src.resolve())
    assert path.is_file() and os.access(path,os.R_OK) and path.stat().st_size==old["bytes"]
total=sum(c["bytes"] for c in expected.values())
assert shutil.disk_usage(repo).free>total+1024**3
assert not dst.exists(),"Destination exists; inspect the receipt before reusing it"
dst.mkdir(parents=True)
receipt={"status":"copying","source_directory":str(src),"destination_directory":str(dst),"training_commit":manifest["training_commit"],
         "handoff_commit":manifest["handoff_commit"],"total_bytes":total,"proxies_used":False,"checkpoints":[]}
receipt_path=audit_root/"checkpoint_import_receipt.json"
write_json(receipt_path,receipt)
started=time.monotonic()
try:
    for name,old in sorted(expected.items()):
        source=src/name;target=dst/name;target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(".pt.partial")
        source_hash=hashlib.sha256();count=0;tick=time.monotonic()
        with source.open("rb") as reader,temporary.open("xb") as writer:
            while True:
                block=reader.read(8*1024**2)
                if not block:break
                source_hash.update(block);writer.write(block);count+=len(block)
            writer.flush();os.fsync(writer.fileno())
        copy_seconds=time.monotonic()-tick
        assert count==old["bytes"] and source_hash.hexdigest()==old["sha256"]
        tick=time.monotonic();actual=sha256_file(temporary);verification_seconds=time.monotonic()-tick
        assert actual==old["sha256"]
        temporary.rename(target)
        item={"scene":old["scene"],"method":old["method"],"file":name,"bytes":count,"sha256":actual,
              "source_path":str(source),"local_HPC_path":str(target),"source_hash_verified":True,"copy_hash_verified":True,
              "copy_seconds":copy_seconds,"verification_seconds":verification_seconds,
              "companion_case_path":str(audit_root/"archive/cases"/old["scene"]/old["method"]),
              "original_dataset_manifest":str(audit_root/"archive/datasets"/old["scene"]/"manifest.json")}
        receipt["checkpoints"].append(item);write_json(receipt_path,receipt)
        print(json.dumps({"verified":len(receipt["checkpoints"]),"total":9,"file":name,"bytes":count}),flush=True)
    for name in ["SHA256SUMS","manifest.json"]:shutil.copyfile(src/name,dst/name)
    baseline=json.loads((repo/"runs/part2_formal/Re10k-1/full_sparse_7000_v1/status.json").read_text())
    assert source_version()["package_source_sha256"]==baseline["source"]["package_source_sha256"]
    receipt.update(status="completed",completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
                   copied_bytes=sum(c["bytes"] for c in receipt["checkpoints"]),source_and_copy_sha256_verified=9,part2_source_unchanged=True)
    write_json(receipt_path,receipt);write_json(dst.parent/"import_receipt.json",receipt)
    text=["# Imported Part 1 checkpoints","",
          "All nine 7000-step checkpoint files are copied from the teammate's shared directory and independently SHA256-verified.",
          f"Training source: {manifest['training_commit']}. Handoff commit: {manifest['handoff_commit']}.",
          "Layout: checkpoints/<scene>/<method>/step_007000.pt.",
          f"Original initialization, configuration, metrics and dataset manifests: {audit_root/'archive'}.",
          "Checkpoints bind the original source and dataset manifest; merged main and Part 2's manifest metadata are not transparent resume substitutes.",
          "No checkpoint was loaded or training resumed during import; Part 2 source and experiments are preserved.",
          "See import_receipt.json for all source paths, copied paths, hashes, sizes and timings.",""]
    (dst.parent/"README.md").write_text("\n".join(text))
    print(json.dumps({k:v for k,v in receipt.items() if k!="checkpoints"},indent=2),flush=True)
except Exception as exc:
    receipt.update(status="failed",error=str(exc));write_json(receipt_path,receipt);raise
