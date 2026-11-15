"""Create a readable report and unaltered-reference/render comparison panel."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
from p4.common import write_json

p=argparse.ArgumentParser()
p.add_argument("--run",default="runs/part2_formal/Re10k-1/full_sparse_7000_v1")
args=p.parse_args(); root=Path(args.run)
state=json.loads((root/"status.json").read_text())
assert state["status"]=="completed"
rows=state["results"]
for row in rows:
    case=root/row["condition"]
    training=[json.loads(line) for line in (case/"train/train.jsonl").read_text().splitlines()]
    assert training[-1]["step"]==7000
    row["training_seconds"]=training[-1]["training_seconds"]
    row["final_gaussians"]=training[-1]["num_gaussians"]
    row["training_peak_gpu_gib"]=max(x["peak_cuda_gib"] for x in training)
write_json(root/"comparison.json",rows)
text=["# Re10k-1 Part 2: 7000-step Full / Sparse comparison","",
      "Both conditions were independently initialized and trained on this HPC.",
      "VGGT without BA; all training views participate jointly in every global attention layer.",
      "The offload backend preserves FP32 weights/residuals and BF16 autocast; it uses full K/V for every query.",
      "Seed 42; identical 7000-step schedule; 100,000 initial-point cap; the same 35 held-out RGB frames.","",
      "| Condition | Train views | PSNR (dB) | SSIM | LPIPS | Paired ATE |",
      "| --- | ---: | ---: | ---: | ---: | ---: |"]
for row in rows:
    text.append(f"| {row['condition']} | {row['train_frames']} | {row['psnr']:.6f} | {row['ssim']:.6f} | {row['lpips']:.6f} | {row['paired_ate_rmse']:.6f} |")
text += ["","ATE is in reference-trajectory units, not asserted meters. Both ATE fits use the same 10 sparse TRAIN frame IDs.",
         "Rendering uses one global Sim3 fitted on the available training correspondences for each condition; held-out poses are never refined.","",
         "## Audits","",
         "- Reference quaternion ordering, c2w convention, intrinsic scaling and normalization projection checks passed.",
         "- OpenCV projection disagreement: 0.0000112 px; normalization projection disagreement: 0.0000549 px.",
         "- Original source ZIP SHA256 matches B's published value; all 1,598 source files match the HPC shared directory.",
         "- Repeating preprocessing with B's published package versions produces exactly the same 784 HPC PNGs and reference arrays.",
         "- B's committed prepared hashes still differ. B's actual manifest/reference/PNG artifacts were unavailable; the cause cannot be established from aggregate hashes.",
         "- No B initialization, checkpoint or score is reused. This comparison is internally consistent and must not silently be merged with B's existing rows.",
         "- Offload versus native validation: max camera rotation difference 0.006115 degrees, max focal difference 0.081016%, depth median relative difference 0.019030%.",
         "- All reconstruction inputs are training RGB. Reference cameras are used only for data calibration/held-out evaluation as documented in the shared protocol.","",
         "## Artifacts","",f"Run: {root.resolve()}",
         "status.json records commands, execution times and source fingerprint.",
         "Each condition contains init/scene.npz, train/checkpoints/step_007000.pt, renders/, metrics.json.",
         "preview.jpg shows fixed frames 00000, 00136 and 00272; it is a preview only and was not used to select metrics."]
(root/"report.md").write_text("\n".join(text)+"\n")
metrics=json.loads((root/"full/metrics.json").read_text())
frames=[metrics["per_frame"][i] for i in [0,17,34]]
canvas=Image.new("RGB",(768,840),"white"); draw=ImageDraw.Draw(canvas)
sources=[("REFERENCE",Path("data/prepared/Re10k-1")),
         ("FULL 244 / 7000",root/"full/renders"),("SPARSE 10 / 7000",root/"sparse/renders")]
for j,(label,source) in enumerate(sources):
    for i,frame in enumerate(frames):
        im=Image.open(source/frame["image"]).convert("RGB")
        assert im.size==(256,256)
        canvas.paste(im,(256*i,280*j+24))
        draw.text((256*i+4,280*j+5),label+" "+frame["id"],fill="black")
canvas.save(root/"preview.jpg",quality=85)
print(json.dumps(rows,indent=2))
