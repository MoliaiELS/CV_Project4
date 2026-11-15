import hashlib,json,zipfile
from pathlib import Path
from p4.common import sha256_file,write_json
archive=Path("/data/disk2/home/zguo400/p4-bootstrap/source_archive_from_local.zip")
published=json.loads(Path("results/data_audit/preparation.json").read_text())
h=sha256_file(archive)
assert h==published["archive_sha256"],(h,published["archive_sha256"])
root=Path("/data/disk1/shared/aiaa3201/project4_3d")
count=0
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        if info.is_dir():continue
        rel=Path(info.filename).relative_to("project4_3d")
        assert ".." not in rel.parts and not rel.is_absolute()
        expected=hashlib.sha256(z.read(info)).hexdigest()
        assert sha256_file(root/rel)==expected,str(rel)
        count+=1
out={"archive_sha256":h,"matches_published_B_source_archive":True,"all_archive_files_match_HPC_shared_source":True,"files_checked":count}
write_json("results/hpc_setup/source_archive_identity.json",out)
print(json.dumps(out))
