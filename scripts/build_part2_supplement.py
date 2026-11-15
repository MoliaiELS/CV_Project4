"""Build reproducible Part 2 figures, analysis and a shared-paper LaTeX section."""
import csv,json,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageOps,ImageFont
from matplotlib.font_manager import findfont
from p4.common import sha256_file,source_version,write_json
load=lambda p:json.loads(Path(p).read_text())
out=Path("results/part2_supplement_v1");fig=out/"figures";fig.mkdir(exist_ok=True,parents=True)
conv=load(out/"convergence_status.json");diag=load(out/"diagnostics.json");bench=load(out/"fast_initialization_benchmark.json")
replay=load(out/"part1_checkpoint_replay.json")
assert conv["status"]==diag["status"]==bench["status"]==replay["status"]=="completed"
rows=load("results/part2_formal_7000/comparison.json")["rows"]
scenes=["Re10k-1","Waymo-405841","Dl3dv-2"];colors={"full":"#2563a6","sparse":"#d86b25"}
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
def save(f,name):
    for suffix in ["png","svg"]:
        target=fig/(name+"."+suffix)
        f.savefig(target,dpi=180,bbox_inches="tight",facecolor="white")
        if suffix=="svg":target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines())+"\n")
    plt.close(f)
def series(scene,kind):return sorted([p for p in conv["points"] if p["scene"]==scene and p["condition"]==kind],key=lambda p:p["step"])
bconv=list(csv.DictReader(Path("results/hpc_setup/part1_handoff_901c5ae/published/results/part1-final/convergence.csv").open()))
for metrics,name in [(["psnr"],"convergence_psnr"),(["psnr","ssim","lpips"],"convergence_all_metrics")]:
    f,axes=plt.subplots(len(metrics),3,figsize=(10.5,3.1*len(metrics)),squeeze=False,constrained_layout=True)
    for j,scene in enumerate(scenes):
        for i,metric in enumerate(metrics):
            ax=axes[i,j]
            for kind in ["full","sparse"]:
                s=series(scene,kind);ax.plot([r["step"] for r in s],[r[metric] for r in s],"-o",ms=4,color=colors[kind],label="Part 2 "+kind.title())
            b=sorted([r for r in bconv if r["scene"]==scene and r["method"]=="vggt"],key=lambda r:int(r["step"]))
            ax.plot([int(r["step"]) for r in b],[float(r[metric]) for r in b],"--",lw=1.4,color="#60836b",label="Part 1 VGGT Full (audit)")
            ax.set(title=scene,xlabel="Training iteration",ylabel=metric.upper()+(" (dB)" if metric=="psnr" else ""));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=10);save(f,name)
f,axes=plt.subplots(2,3,figsize=(10.5,7),constrained_layout=True)
for j,scene in enumerate(scenes):
    with np.load(out/(scene+"_paired_trajectories.npz")) as d:
        center=d["reference"].mean(0);_,_,basis=np.linalg.svd(d["reference"]-center,full_matrices=False)
        ref=(d["reference"]-center)@basis[:2].T;ax=axes[0,j]
        ax.plot(ref[:,0],ref[:,1],"o-",color="#555555",label="Reference",ms=4,lw=1)
        for kind in ["full","sparse"]:
            xy=(d[kind]-center)@basis[:2].T
            ax.plot(xy[:,0],xy[:,1],"o--",color=colors[kind],label=kind.title(),ms=3,lw=1)
            axes[1,j].plot(np.arange(len(d["names"])),d[kind+"_error"],"o-",ms=3,color=colors[kind],label=kind.title())
        ax.set(title=scene,xlabel="Reference trajectory PC1",ylabel="PC2 (reference units)");ax.axis("equal");ax.grid(alpha=.2)
        axes[1,j].set(xlabel="Common training-frame index",ylabel="Center error (reference units)");axes[1,j].grid(alpha=.2)
axes[0,0].legend(fontsize=10);axes[1,0].legend(fontsize=10);save(f,"paired_trajectories")
runtime=[]
for scene in scenes:
    state=load(Path("runs/part2_formal")/scene/"full_sparse_7000_v1/status.json")
    for kind in ["full","sparse"]:
        stages={s["name"]:s["seconds"] for s in state["stages"]}
        runtime.append({"scene":scene,"condition":kind,**{s:stages[kind+"_"+s] for s in ["initialize","train","render","evaluate"]}})
with (out/"runtime.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(runtime[0]));w.writeheader();w.writerows(runtime)
f,axes=plt.subplots(1,3,figsize=(10.5,3.5),constrained_layout=True)
for j,scene in enumerate(scenes):
    r=[r for r in runtime if r["scene"]==scene];bottom=np.zeros(2)
    for key,color in zip(["initialize","train","render","evaluate"],["#34495e","#3b82b6","#8cbdd2","#e2a34e"]):
        values=np.array([z[key] for z in r]);axes[j].bar(["Full","Sparse"],values,bottom=bottom,label=key.title(),color=color);bottom+=values
    axes[j].set(title=scene,ylabel="Historical stage wall time (s)")
axes[0].legend(fontsize=10);save(f,"runtime_breakdown")
f,axes=plt.subplots(1,2,figsize=(10.5,3.5),constrained_layout=True);x=np.arange(3)
for j,(fields,ylabel) in enumerate([(["historical_cpu_disk_seconds","cpu_feature_seconds"],"Initialization time (s)"),(["historical_peak_gpu_gib","cpu_feature_peak_gpu_gib"],"Peak allocated GPU memory (GiB)")]):
    for k,(field,label,color) in enumerate(zip(fields,["Original CPU/disk (historical)","CPU head features (new)"],["#50647d","#2c9973"])):
        axes[j].bar(x+(k-.5)*.35,[r[field] for r in bench["rows"]],width=.35,label=label,color=color)
    axes[j].set_xticks(x,scenes,rotation=12);axes[j].legend(fontsize=10);axes[j].set_ylabel(ylabel);axes[j].set_ylim(0,1.28*max(r[field] for field in fields for r in bench["rows"]))
save(f,"initialization_tradeoff")
coverage={(r["scene"],r["condition"]):r for r in diag["coverage"]}
f,ax=plt.subplots(figsize=(9,3.5),constrained_layout=True)
for k,kind in enumerate(["full","sparse"]):ax.bar(x+(k-.5)*.35,[100*coverage[(s,kind)]["coverage_alpha_0_5"] for s in scenes],width=.35,label=kind.title(),color=colors[kind])
ax.set_xticks(x,scenes);ax.set_ylim(0,101);ax.set_ylabel("Projected support, alpha >= 0.5 (%)");ax.legend(loc="lower center",bbox_to_anchor=(.5,1.01),ncol=2,frameon=False);save(f,"opacity_coverage")
canvas=Image.new("RGB",(1600,810),"white");draw=ImageDraw.Draw(canvas);label_font=ImageFont.truetype(findfont("DejaVu Sans"),16);failure=[]
for j,scene in enumerate(scenes):
    root=Path("runs/part2_formal")/scene/"full_sparse_7000_v1"
    fm=load(root/"full/metrics.json")["per_frame"];sm=load(root/"sparse/metrics.json")["per_frame"]
    i=int(np.argmax([a["psnr"]-b["psnr"] for a,b in zip(fm,sm)]));a,b=fm[i],sm[i];name=a["image"]
    paths=[Path("data/prepared")/scene/name,root/"full/renders"/name,root/"sparse/renders"/name,Path("runs/part2_supplement_v1/coverage")/scene/"full"/name,Path("runs/part2_supplement_v1/coverage")/scene/"sparse"/name]
    labels=["Reference",f"Full {a['psnr']:.2f} dB",f"Sparse {b['psnr']:.2f} dB","Full alpha (white=1)","Sparse alpha (white=1)"]
    for k,(path,label) in enumerate(zip(paths,labels)):
        with Image.open(path) as im:tile=ImageOps.contain(im.convert("RGB"),(316,224),Image.Resampling.LANCZOS)
        canvas.paste(tile,(k*320+(320-tile.width)//2,j*270+42+(224-tile.height)//2))
        draw.text((k*320+4,j*270+6),scene+" / "+a["id"],fill="black",font=label_font);draw.text((k*320+4,j*270+23),label,fill="black",font=label_font)
    failure.append({"scene":scene,"image":name,"selection":"largest Full-minus-Sparse PSNR gap across all test frames","full_psnr":a["psnr"],"sparse_psnr":b["psnr"],"gap_db":a["psnr"]-b["psnr"]})
canvas.save(fig/"failure_cases.png");write_json(out/"failure_cases.json",failure)
peaks=[]
for scene in scenes:
    for kind in ["full","sparse"]:
        s=series(scene,kind);peak=max(s,key=lambda p:p["psnr"]);last=s[-1]
        peaks.append({"scene":scene,"condition":kind,"highest_sampled_psnr_step":peak["step"],"highest_sampled_psnr":peak["psnr"],"final_psnr":last["psnr"],"peak_minus_final_db":peak["psnr"]-last["psnr"]})
write_json(out/"convergence_observations.json",peaks)
lines=["# Part 2 enhanced analysis","",
"All six original 7000-step results are preserved. No new scene training or test-based checkpoint selection was performed.",
"Custom VGGT-initialized unposed reconstruction: VGGT estimates cameras and geometry only from permitted training RGB; vanilla 3DGS keeps those cameras fixed. This is not an i3dgs reproduction or SLAM.","",
"## Frozen comparison","",
"| Scene | Input | Train / test | PSNR | SSIM | LPIPS | Paired ATE |","| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
for r in rows:lines.append(f"| {r['scene']} | {r['condition']} | {r['train_frames']} / {r['test_frames']} | {r['psnr']:.3f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {r['paired_ate_rmse']:.5f} |")
lines+=["","Full has higher PSNR and SSIM in all scenes. Dl3dv has the largest PSNR gain (3.13 dB); Waymo Sparse has lower LPIPS on all 25 test views.",
"Seed 42, 7000 updates; sparse original positions 1, 1+k, ... use k=10 for Waymo and k=30 otherwise, excluding fixed held-out indices.","",
"A fixed update budget gives different average exposures per image: Re10k Full/Sparse 28.7/700, Waymo 40.2/350, and Dl3dv 26.2/636.4. This fixes update count, not epoch count; image repetition is not isolated as a separate factor.","",
"## Convergence","",
"All 30 condition/iteration points at 500/1000/2000/4000/7000 steps cover 990 per-frame evaluations. Twenty-four intermediate points were evaluated from saved checkpoints; six final points keep the original measurements.",
"The 7000-step endpoint was fixed before evaluation. Highest sampled test PSNR is diagnostic only and is not used for model selection.","",
"| Scene | Input | Highest sampled PSNR step | Peak minus final |","| --- | --- | ---: | ---: |"]
for p in peaks:lines.append(f"| {p['scene']} | {p['condition']} | {p['highest_sampled_psnr_step']} | {p['peak_minus_final_db']:.3f} dB |")
lines+=["","Non-monotonic curves alone do not identify a cause; fixed camera bias, scene coverage and fitting behavior can contribute.",
"![PSNR convergence](figures/convergence_psnr.png)","",
"## Cameras and visibility support","",
"Paired ATE and trajectory figures align both conditions on the same sparse TRAIN IDs. The separate all-train ATE matches the support used for rendering alignment and Part 1's reported convention. Units are reference-trajectory units, not assumed meters.",
"Camera optimization is disabled: ATE is constant during GS training. An ATE-versus-GS-iteration convergence plot would be misleading.",
"Current Part 1 image hashes, splits and reference cameras match our data. Re10k's roughly 20% focal overestimate also appears in the independent Part 1 VGGT initialization; reference intrinsics are not substituted to repair it.",
"![Paired trajectories](figures/paired_trajectories.png)","",
"Alpha coverage measures projected Gaussian support, not true 3D completeness or accuracy. Opaque incorrect geometry can have high coverage.","",
"| Scene | Input | Alpha >= 0.1 | Alpha >= 0.5 | Alpha >= 0.9 |","| --- | --- | ---: | ---: | ---: |"]
for r in diag["coverage"]:lines.append(f"| {r['scene']} | {r['condition']} | {r['coverage_alpha_0_1']:.2%} | {r['coverage_alpha_0_5']:.2%} | {r['coverage_alpha_0_9']:.2%} |")
lines+=["","Failure panels select the largest Full-minus-Sparse PSNR gap per scene by a declared rule. All test frames remain in metrics; first/middle/last previews remain in the formal runs.",
"![Failures and alpha](figures/failure_cases.png)","",
"In these selected views, Re10k shows distorted cabinet/room boundaries in both reconstructions. Waymo Full is smoother around cars and roadside structures, while Sparse retains sharper-looking edges alongside geometric artifacts; this does not isolate the cause of the LPIPS ranking. Dl3dv Sparse shows fragmented foliage and a dark uncovered region at the lower-right ground boundary, whereas Full is more coherent. Alpha panels use white=1 and black=0.","",
"## Initialization optimization","",
"The adapter imports CPU-head-feature offload from pinned main 901c5ae without changing the checkpoint-bound p4 package. All views, weights, preprocessing, seed and filtering are preserved. The CPU/disk route remains available for lower memory.","",
"| Scene | Original init (s) | Feature offload (s) | Observed ratio | New peak GPU GiB | Max rotation delta |","| --- | ---: | ---: | ---: | ---: | ---: |"]
for r in bench["rows"]:lines.append(f"| {r['scene']} | {r['historical_cpu_disk_seconds']:.1f} | {r['cpu_feature_seconds']:.1f} | {r['observed_time_ratio']:.2f}x | {r['cpu_feature_peak_gpu_gib']:.2f} | {r['max_rotation_difference_degrees']:.6f} deg |")
lines+=["",bench["interpretation"],"These are initialization-only trials; no improved final rendering score is claimed from memory scheduling.",
"![Time and memory](figures/initialization_tradeoff.png)","",
"## Reuse and limitations","",
"Part 1's nine checkpoints are imported and SHA256-verified. Its independent VGGT Full scores differ by less than 0.09 dB from our Full scores. B's Full curves and recomputed common-frame ATE are audit comparisons; our primary paired runs remain unchanged.",
"Part 1 used an RTX 4090; these Part 2 runs used an RTX A6000. Their wall times are not a controlled cross-hardware comparison.",
"Checkpoint restoration requires the original package fingerprint and manifest. The upstream Adam CPU-step restoration fix is recorded for future continuation work; it is not injected into the package used for these frozen evaluations.",
"One seed and three scenes do not support population-level statistical claims. There is no true 3D reference surface for completeness scoring. No Part 3 generation was run.","",
"## Reproduce on HPC","",
"Activate p4-bootstrap/direct.env and envs/project4, then run these module commands from the preserved Part 2 checkout:",
"- python -m scripts.part2_convergence","- python -m scripts.part2_diagnostics","- python -m scripts.part2_fast_initialize --benchmark","- python -m scripts.part1_checkpoint_replay","- python -m scripts.build_part2_supplement",
"HF_HUB_OFFLINE=1 is used; no proxies or model downloads are required. Large checkpoints and intermediate renders remain under runs/.",""]
lines += ["## Checkpoint replay acceptance", "", "All three imported Part 1 VGGT Full checkpoints were re-rendered on the course HPC using the exact historical package and original manifest. The primary Part 2 scores were not replaced.", "", "| Scene | Replayed PSNR | Difference from archived PSNR |", "| --- | ---: | ---: |"]
for r in replay["rows"]:lines.append(f"| {r['scene']} | {r['replayed_metrics']['psnr']:.6f} | {r['replay_minus_original']['psnr']:+.6f} dB |")
lines += ["", "## Common-frame camera breakdown", "", "| Scene | Input | Paired ATE | Mean orientation error (deg) | Median fx ratio | Median fy ratio | All-train ATE |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
for r in diag["camera_diagnostics"]:lines.append(f"| {r['scene']} | {r['condition']} | {r['paired_ate_rmse']:.5f} | {r['paired_rotation_mean_degrees']:.3f} | {r['paired_focal_ratio_fx']:.4f} | {r['paired_focal_ratio_fy']:.4f} | {r['all_available_train_ate']:.5f} |")
lines += ["", "Waymo illustrates different error rankings: Sparse has lower center ATE, but a larger paired mean orientation error (2.214 versus 1.053 degrees). This is an observed association, not an isolated causal attribution.", ""]
lines += ["## Complete metric curves and runtime", "", "![PSNR, SSIM and LPIPS](figures/convergence_all_metrics.png)", "", "![Historical stage runtime](figures/runtime_breakdown.png)", "", "Historical elapsed times include each stage startup and local I/O, not only GPU kernel time. The initialization benchmark above is separate from these original runs.", "", "![Projected opacity support](figures/opacity_coverage.png)", ""]
(out/"report.md").write_text("\n".join(lines).rstrip()+"\n")
report=Path("report");report.mkdir(exist_ok=True)
tex=[r"\section{Unposed full-versus-sparse reconstruction}",r"\label{sec:part2}",
r"We implement a custom VGGT-initialized unposed pipeline~\cite{wang2025vggt}, followed by fixed-camera 3D Gaussian Splatting~\cite{kerbl2023gaussians}. Full and sparse branches independently estimate cameras and geometry from their permitted training RGB. Reference cameras are used only for evaluation.",
r"\paragraph{Protocol.} We retain every tenth original Waymo frame and every thirtieth DL3DV/RealEstate10K frame, excluding fixed held-out indices. Full/sparse training counts are 174/20, 267/11 and 244/10; test counts are 25, 39 and 35. Both conditions use 7,000 updates, seed 42, a 100k initial-point cap and no camera optimization.",
r"\begin{table*}[t]\centering\small",r"\begin{tabular}{llrrrrr}\toprule Scene & Input & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & Paired ATE $\downarrow$ & Views \\ \midrule"]
for r in rows:tex.append(f"{r['scene']} & {r['condition'].title()} & {r['psnr']:.3f} & {r['ssim']:.4f} & {r['lpips']:.4f} & {r['paired_ate_rmse']:.5f} & {r['train_frames']} \\\\")
tex += [r"\bottomrule\end{tabular}\caption{Fixed 7,000-step comparison. Paired ATE fits each estimate on the same sparse training camera IDs. Reference-trajectory units differ across scenes and are not averaged.}\label{tab:part2_main}\end{table*}",
r"\paragraph{Results.} Full improves PSNR and SSIM in all scenes, with a 3.13 dB PSNR gain on DL3DV. Waymo Sparse has lower LPIPS on all 25 test frames. Sparse has lower paired camera-center ATE in every scene; this does not establish better intrinsics, orientation, scene support or novel-view quality.",
r"\paragraph{Convergence.} We evaluate saved checkpoints at 500, 1,000, 2,000, 4,000 and 7,000 updates and retain the preselected final endpoint. Intermediate test peaks are diagnostic, not model-selection criteria. Fixed cameras imply constant ATE throughout GS training.",
r"\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{convergence_psnr.png}\caption{PSNR at five fixed steps. Dashed Part 1 VGGT Full curves are independent-run audits; the primary comparison remains the Part 2 pair.}\label{fig:part2_convergence}\end{figure*}",
r"\paragraph{Failure analysis.} We compare aligned trajectories and accumulated Gaussian alpha at the same test views. Alpha is a projected-support proxy, not true 3D completeness: opaque incorrect surfaces may score highly. Failure panels select the largest Full-minus-Sparse PSNR gap with all test frames retained in scoring. Re10k's roughly 20\% focal bias is shared by the independent Part 1 initialization and is not corrected using reference parameters.",
r"\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{failure_cases.png}\caption{Largest PSNR-gap view per scene and accumulated-alpha maps, supplementing fixed first/middle/last previews.}\label{fig:part2_failures}\end{figure*}",
r"\paragraph{Engineering and limitations.} A separate adapter reuses CPU-head-feature offload and verifies camera parity with the original CPU/disk initialization. Historical/new wall-time ratios are not controlled repeated benchmarks. We preserve checkpoint-bound source fingerprints and original paired results. These observations cover one seed and three scenes; no generative Part 3 gain is claimed.",""]
tex.append("Initialization time changes from " + ", ".join(f"{r['historical_cpu_disk_seconds']:.0f} to {r['cpu_feature_seconds']:.0f} s ({r['scene']})" for r in bench["rows"]) + f"; peak allocated GPU memory is {min(r['cpu_feature_peak_gpu_gib'] for r in bench['rows']):.1f}--{max(r['cpu_feature_peak_gpu_gib'] for r in bench['rows']):.1f} GiB. The exported cameras and initial point clouds pass parity checks.\n")
(report/"part2.tex").write_text("\n".join(tex))
(report/"part2_integration.md").write_text("Include part2.tex in the shared CVPR paper. Add ../results/part2_supplement_v1/figures/ to the graphics path when building from report/. Reuse kerbl2023gaussians and wang2025vggt from the Part 1 bibliography. Requires booktabs and graphicx. This is a LaTeX fragment, not a compiled final submission.\n")
(out/"midterm_part2_notes.md").write_text("# Member A: two-page content\n\n## Page 1: method\nShow the custom VGGT no-BA -> fixed-camera 3DGS pipeline and the six-condition table. State strides 10/30/30, seed 42 and 7000 steps; reference cameras are evaluation-only. Full and Sparse were independently initialized and trained.\n\n## Page 2: results\nShow convergence_psnr.png and either failure_cases.png or paired_trajectories.png. Full improves PSNR/SSIM in every scene, most strongly on DL3DV (+3.13 dB). Waymo Sparse improves LPIPS. Lower sparse trajectory error does not guarantee higher rendering quality. Mention non-monotonic test curves, camera bias and visibility support.\n\nPart 3 remains future work: two generative reconstruction rounds with matched real-only continuation controls.\n")
assert len(bench["rows"]) == 3 and all(r["camera_parity_passed"] and r["point_parity_passed"] for r in bench["rows"])
assert len(replay["rows"]) == 3
assert len(conv["points"])==30 and conv["total_per_frame_evaluations"]==990
assert len(load(out/"coverage_per_frame.json"))==198
assert source_version()["package_source_sha256"]==conv["source_sha256"]
assert sha256_file("results/part2_formal_7000/summary_bundle.zip")=="4f06678ebe7243dfa6f18cade1140828d203f77ab6afad5972efa9f1bb5192b0"
write_json(out/"verification.json",{"status":"completed","convergence_points":30,"new_points":24,"frame_evaluations":990,"coverage_renders":198,"fast_parity_passes":3,"Part1_full_checkpoint_replays":3,"original_package_and_bundle_unchanged":True,"new_training_runs":0,"test_based_selection":False,"source":source_version(),"original_rows":rows})
files=[p for p in out.rglob("*") if p.is_file() and p.suffix!=".zip" and p.name!="bundle_manifest.json"]
files += [report/"part2.tex",report/"part2_integration.md"]+[Path("scripts")/n for n in ["part2_convergence.py","part2_diagnostics.py","part2_fast_initialize.py","build_part2_supplement.py","part1_checkpoint_replay.py"]]+[Path("runs/part2_supplement_v1/source/initialize_901c5ae.py")]
write_json(out/"bundle_manifest.json",{str(p):sha256_file(p) for p in files});files.append(out/"bundle_manifest.json")
with zipfile.ZipFile(out/"part2_enhanced_bundle.zip","w",zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,arcname=str(p))
with zipfile.ZipFile(out/"part2_enhanced_bundle.zip") as z:assert z.testzip() is None
print(json.dumps({"status":"completed","directory":str(out.resolve()),"bundle_bytes":(out/"part2_enhanced_bundle.zip").stat().st_size,"figures":len(list(fig.glob("*.png")))},indent=2))
