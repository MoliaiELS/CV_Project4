import json
from pathlib import Path
import numpy as np
import pytest
from p4.common import sha256_file
from p4.sparse import prepare_sparse

def fixture(tmp_path):
    root = tmp_path / "dense"
    (root / "images").mkdir(parents=True)
    frames = []
    for i in range(81):
        path = root / "images" / f"{i:05d}.png"
        path.write_bytes(f"frozen-image-{i}".encode())
        frames.append({"id":f"{i:05d}.png", "image":f"images/{i:05d}.png",
                       "split":"test" if i % 8 == 0 else "train", "sha256":sha256_file(path),
                       "width":256,"height":256})
    m={"schema_version":1,"scene":"fixture","camera_convention":"opencv_c2w","max_size":960,
       "preprocessing":"test-fixture","frames":frames,"split_hash":"parent","reference_kind":"reference"}
    (root/"manifest.json").write_text(json.dumps(m))
    np.savez(root/"reference.npz", names=[f["image"] for f in frames],
             c2w=np.broadcast_to(np.eye(4),(81,4,4)),K=np.broadcast_to(np.eye(3),(81,3,3)))
    return root,m

def test_original_stride_and_test_invariance(tmp_path):
    root,parent=fixture(tmp_path)
    before=(root/"manifest.json").read_bytes()
    child=prepare_sparse(root,tmp_path/"sparse",30,1)
    assert child["split_source"]["selected_original_positions"]==[1,31,61]
    assert [f for f in child["frames"] if f["split"]=="test"]==[f for f in parent["frames"] if f["split"]=="test"]
    assert [f["id"] for f in child["frames"] if f["split"]=="train"]==["00001.png","00031.png","00061.png"]
    assert len(list((tmp_path/"sparse/images").glob("*.png")))==14
    assert (root/"manifest.json").read_bytes()==before
    with np.load(tmp_path/"sparse/reference.npz") as r:
        assert r["names"].tolist()==[f["image"] for f in child["frames"]]
    assert child["split_hash"] != parent["split_hash"]

def test_reject_test_collision(tmp_path):
    root,_=fixture(tmp_path)
    with pytest.raises(ValueError,match="overlaps"):
        prepare_sparse(root,tmp_path/"sparse",30,0)
    assert not (tmp_path/"sparse").exists()

def test_refuse_changed_input(tmp_path):
    root,_=fixture(tmp_path)
    (root/"images/00001.png").write_bytes(b"changed")
    with pytest.raises(ValueError,match="RGB changed"):
        prepare_sparse(root,tmp_path/"sparse",30,1)
    assert not (tmp_path/"sparse").exists()

def test_sampling_independent_of_reference(tmp_path):
    root,_=fixture(tmp_path)
    first=prepare_sparse(root,tmp_path/"first",30,1)
    (root/"reference.npz").unlink()
    second=prepare_sparse(root,tmp_path/"second",30,1)
    assert first["split_source"]["selected_original_frame_ids"]==second["split_source"]["selected_original_frame_ids"]
    assert first["split_hash"]==second["split_hash"]

def test_preserve_existing_output(tmp_path):
    root,_=fixture(tmp_path)
    prepare_sparse(root,tmp_path/"sparse",30,1)
    with pytest.raises(FileExistsError):
        prepare_sparse(root,tmp_path/"sparse",30,1)
