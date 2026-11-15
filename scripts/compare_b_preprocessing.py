"""Compare HPC preparation with a reproduction using B's published packages."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from p4.common import sha256_file, write_json

published = json.loads(Path("results/data_audit/preparation.json").read_text())
rows = []
for expected in published["scenes"]:
    scene = expected["scene"]
    a, b = (Path(root) / scene for root in ["data/prepared", "data/prepared_b_versions"])
    ma, mb = (json.loads((p / "manifest.json").read_text()) for p in [a, b])
    changes, pixels, pixels_changed, abs_sum, values, max_delta = [], [], 0, 0, 0, 0
    assert len(ma["frames"]) == len(mb["frames"])
    for fa, fb in zip(ma["frames"], mb["frames"]):
        assert {k:v for k,v in fa.items() if k!="sha256"} == {k:v for k,v in fb.items() if k!="sha256"}
        ia, ib = (np.asarray(Image.open(p / f["image"]).convert("RGB"), dtype=np.int16) for p,f in [(a,fa),(b,fb)])
        delta = np.abs(ia-ib)
        if fa["sha256"] != fb["sha256"]: changes.append(fa["id"])
        if np.any(delta): pixels.append(fa["id"])
        pixels_changed += int(np.any(delta, axis=2).sum())
        abs_sum += int(delta.sum()); values += delta.size
        max_delta = max(max_delta, int(delta.max()))
    with np.load(a/"reference.npz",allow_pickle=False) as ra, np.load(b/"reference.npz",allow_pickle=False) as rb:
        assert np.array_equal(ra["names"],rb["names"])
        reference_diffs = {k:float(np.max(np.abs(ra[k]-rb[k]))) for k in ["c2w","K"]}
        for k in reference_diffs: assert reference_diffs[k] < 1e-12
    rows.append({"scene":scene,"frames":len(ma["frames"]),"frame_assignment_and_dimensions_identical":True,
        "hpc_split_hash":ma["split_hash"],"b_versions_split_hash":mb["split_hash"],
        "published_b_split_hash":expected["split_hash"],"b_versions_match_published_b":mb["split_hash"]==expected["split_hash"],
        "b_versions_reference_matches_published_b":sha256_file(b/"reference.npz")==expected["reference_sha256"],
        "png_byte_changed_frames":len(changes),"decoded_rgb_changed_frames":len(pixels),"changed_pixels":pixels_changed,
        "max_channel_delta_0_255":max_delta,"mean_absolute_channel_delta_0_255":abs_sum/values,
        "reference_numeric_max_abs_difference":reference_diffs})
report={"reproduction_versions":published["preprocessing_versions"],"scenes":rows}
write_json("results/hpc_setup/b_preprocessing_comparison.json",report)
print(json.dumps(report,indent=2))
