"""Run a self-contained 7k scene Full/Sparse comparison after audit gates."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from p4.common import write_json,sha256_file,source_version
from p4.data import load_manifest,load_scene
from p4.geometry import align_training_cameras

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",choices=["Re10k-1","Waymo-405841","Dl3dv-2"],default="Re10k-1")
    p.add_argument("--output")
    args=p.parse_args()
    root=Path(args.output or f"runs/part2_formal/{args.scene}/full_sparse_7000_v1")
    if root.exists():raise FileExistsError("Use a new output directory")
    load=lambda path:json.loads(Path(path).read_text())
    assert load("results/hpc_setup/camera_audit.json")["status"]=="passed"
    source=load("results/hpc_setup/source_archive_identity.json")
    assert source["matches_published_B_source_archive"] and source["all_archive_files_match_HPC_shared_source"]
    comparison=load("results/hpc_setup/b_preprocessing_comparison.json")
    assert all(x["decoded_rgb_changed_frames"]==0 and max(x["reference_numeric_max_abs_difference"].values())<1e-12 for x in comparison["scenes"])
    assert load("runs/offload_parity_01/parity.json")["status"]=="passed"
    assert "48 passed" in Path("/data/disk2/home/zguo400/p4-bootstrap/cpu_tests_02.log").read_text()
    baseline=Path("runs/part2_formal/Re10k-1/full_sparse_7000_v1")
    baseline_state=load(baseline/"status.json")
    assert baseline_state["status"]=="completed"
    assert source_version()["package_source_sha256"]==baseline_state["source"]["package_source_sha256"]
    nf,ns,nt={"Re10k-1":(244,10,35),"Waymo-405841":(174,20,25),"Dl3dv-2":(267,11,39)}[args.scene]
    if args.scene!="Re10k-1":
        preflight=load("results/hpc_setup/remaining_scenes_preflight.json")
        assert preflight["scenes"][args.scene]["status"]=="passed"
        assert preflight["package_source_sha256"]==baseline_state["source"]["package_source_sha256"]
    datasets={"full":Path("data/prepared")/args.scene,"sparse":Path("data/part2_sparse")/args.scene}
    manifests={k:load_manifest(v) for k,v in datasets.items()}
    tests={k:[f for f in m["frames"] if f["split"]=="test"] for k,m in manifests.items()}
    assert tests["full"]==tests["sparse"] and len(tests["full"])==nt
    train={k:{f["image"] for f in m["frames"] if f["split"]=="train"} for k,m in manifests.items()}
    assert len(train["full"])==nf and len(train["sparse"])==ns and train["sparse"]<train["full"]
    env={k:v for k,v in os.environ.items() if k.lower().endswith("_proxy")}
    assert not any(v for k,v in env.items() if k.lower()!="no_proxy")
    root.mkdir(parents=True)
    cfg=load(baseline/"frozen_config.json")
    assert cfg["training"]["steps"]==7000 and cfg["training"]["seed"]==42
    write_json(root/"frozen_config.json",cfg)
    assert sha256_file(root/"frozen_config.json")==sha256_file(baseline/"frozen_config.json")
    state={"status":"running","scene":args.scene,"steps":7000,"seed":42,"proxy_environment":env,
           "stages":[],"source":source_version(),"test_frames":nt,"training_frames":{"full":nf,"sparse":ns},"baseline_run":str(baseline),"frozen_config_sha256":sha256_file(root/"frozen_config.json"),
           "initialization":"VGGT no BA, all training views jointly, CPU/disk offload; 100k-point cap",
           "B_prepared_hash_status":"unresolved; no B checkpoint/result is reused",
           "comparison_basis":"same verified source archive, canonical images and backend on this HPC"}
    write_json(root/"status.json",state)
    def resources(label,cpu_mib,gpu_mib=6000):
        started=time.monotonic()
        while True:
            free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).splitlines()[0])
            mem={l.split(":")[0]:int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()}
            if free>=gpu_mib and mem["MemAvailable"]/1024>=cpu_mib:return
            if time.monotonic()-started>900:raise RuntimeError("Resource headroom unavailable for "+label)
            print(json.dumps({"waiting_resources":label,"gpu_free_mib":free,"ram_available_mib":mem["MemAvailable"]/1024}),flush=True)
            time.sleep(20)
    def run(label,argv):
        assert source_version()["package_source_sha256"]==state["source"]["package_source_sha256"]
        print("START "+label,flush=True)
        start=time.monotonic()
        with (root/(label+".log")).open("w") as log:
            result=subprocess.run([sys.executable]+argv,stdout=log,stderr=subprocess.STDOUT)
        item={"name":label,"seconds":time.monotonic()-start,"returncode":result.returncode,"command":argv}
        state["stages"].append(item);write_json(root/"status.json",state)
        print("DONE "+json.dumps(item),flush=True)
        if result.returncode:raise RuntimeError(label+" failed; see its log")
    try:
        for kind,dataset in datasets.items():
            case=root/kind; case.mkdir()
            resources(kind+"_initialize",12000 if kind=="full" else 6500)
            run(kind+"_initialize",["-m","p4.initialize","--dataset",str(dataset),"--output",str(case/"init"),
                "--method","vggt","--repo","third_party/vggt","--weights","weights/vggt/model.pt",
                "--neural-backend","offload","--seed","42","--max-points","100000","--confidence-quantile","0.5"])
            resources(kind+"_train",5000,10000 if args.scene=="Waymo-405841" else 6000)
            run(kind+"_train",["-m","p4.train","train","--dataset",str(dataset),"--scene",str(case/"init/scene.npz"),
                "--output",str(case/"train"),"--config",str(root/"frozen_config.json")])
            checkpoint=case/"train/checkpoints/step_007000.pt"
            run(kind+"_render",["-m","p4.train","render","--dataset",str(dataset),"--scene",str(case/"init/scene.npz"),
                "--checkpoint",str(checkpoint),"--output",str(case/"renders")])
            run(kind+"_evaluate",["-m","p4.evaluate","renders","--dataset",str(dataset),"--render-dir",str(case/"renders"),
                "--output",str(case/"metrics.json")])
        rows=[]
        with np.load(datasets["full"]/"reference.npz",allow_pickle=False) as ref:
            for kind in datasets:
                case=root/kind; m=load(case/"metrics.json"); init=load(case/"init/scene.json")
                assert m["complete"] and m["test_frames"]==nt and init["registered_frames"]==len(train[kind])
                assert init["all_training_views_joint"] and init["neural_backend"]=="offload"
                sc=load_scene(case/"init/scene.npz")
                paired=align_training_cameras(sc["names"],sc["c2w"],ref["names"],ref["c2w"],train["sparse"])
                rows.append({"condition":kind,"train_frames":len(train[kind]),"test_frames":nt,"steps":7000,
                    **{k:m[k] for k in ["psnr","ssim","lpips"]},"paired_ate_rmse":paired["ate_rmse"],
                    "paired_ate_frame_count":len(paired["train_names"]),"paired_ate_units":"reference trajectory units",
                    "split_hash":m["split_hash"],"weights_sha256":init["weights_sha256"],
                    "checkpoint_sha256":sha256_file(case/"train/checkpoints/step_007000.pt"),
                    "initialization_seconds":init["initialization_seconds"],"initialization_peak_gpu_gib":init["peak_gpu_memory_bytes"]/1024**3})
        assert rows[0]["weights_sha256"]==rows[1]["weights_sha256"]
        assert rows[0]["weights_sha256"]==baseline_state["results"][0]["weights_sha256"]
        state.update(status="completed",results=rows,paired_ate_protocol=f"Fit each estimate to reference using the same {ns} sparse TRAIN camera IDs. Rendering alignment separately uses all available training correspondences for each condition.")
        write_json(root/"status.json",state);write_json(root/"comparison.json",rows)
        print("COMPLETE "+json.dumps(rows),flush=True)
    except Exception as exc:
        state.update(status="failed",error=str(exc));write_json(root/"status.json",state);raise

if __name__=="__main__":main()
