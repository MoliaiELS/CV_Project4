"""Use the pinned Part 1 CPU-feature VGGT route without changing the Part 2 package."""
import argparse,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from p4.common import source_version,sha256_file,write_json
REV="901c5aedfe4627e7af93814e8f37e847237adb7b"
BASE=Path("runs/part2_supplement_v1/fast_initialization")
OUT=Path("results/part2_supplement_v1")
load=lambda p:json.loads(Path(p).read_text())
def initialize(dataset,output):
    assert not any(v for k,v in os.environ.items() if k.lower().endswith("_proxy") and k.lower()!="no_proxy")
    manifest=load(Path(dataset)/"manifest.json");n=sum(f["split"]=="train" for f in manifest["frames"])
    minimum=16000 if n>250 else (14000 if n>200 else 11000)
    start=time.monotonic()
    while True:
        gpu=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).splitlines()[0])
        ram=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))/1024
        if gpu>=30000 and ram>=minimum:break
        if time.monotonic()-start>600:raise RuntimeError("Insufficient headroom for CPU-feature route; original low-memory route remains available")
        print(json.dumps({"waiting_resources":True,"free_gpu_mib":gpu,"available_ram_mib":ram}),flush=True);time.sleep(15)
    snapshot=Path("runs/part2_supplement_v1/source/initialize_901c5ae.py")
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    data=subprocess.check_output(["git","show",REV+":p4/initialize.py"])
    if snapshot.exists():assert snapshot.read_bytes()==data
    else:snapshot.write_bytes(data)
    spec=importlib.util.spec_from_file_location("p4_part1_feature_initializer",snapshot)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run_initializer(dataset=str(dataset),output=str(output),method="vggt",
        repo="third_party/vggt",weights="weights/vggt/model.pt",seed=42,max_points=100000,
        confidence_quantile=0.5,vggt_offload_features=True,neural_backend="native")
    write_json(Path(output)/"adapter_provenance.json",{"upstream_commit":REV,"upstream_file":"p4/initialize.py",
               "upstream_file_sha256":sha256_file(snapshot),"adapter_sha256":sha256_file(__file__),
               "base_package":source_version(),"entrypoint":"run_initializer(vggt_offload_features=True, neural_backend=native)",
               "all_training_views_joint":True,"no_bundle_adjustment":True,"proxies_used":False,
               "resource_headroom_before_start":{"gpu_mib":gpu,"ram_mib":ram}})
    return result
def benchmark():
    BASE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    state={"status":"running","upstream_commit":REV,"rows":[],
           "interpretation":"One new same-GPU initialization timing compared with historical formal initialization; scheduling/load differ, so ratios are observed rather than controlled repeated speedups."}
    write_json(OUT/"fast_initialization_benchmark.json",state)
    try:
        for scene in ["Re10k-1","Waymo-405841","Dl3dv-2"]:
            dest=BASE/scene
            if not (dest/"scene.json").exists() or load(dest/"scene.json")["status"]!="completed":
                with (BASE/(scene+".log")).open("w") as log:
                    r=subprocess.run([sys.executable,"-m","scripts.part2_fast_initialize","--dataset","data/prepared/"+scene,"--output",str(dest)],stdout=log,stderr=subprocess.STDOUT)
                if r.returncode:raise RuntimeError(scene+" fast initialization failed")
            old_dir=Path("runs/part2_formal")/scene/"full_sparse_7000_v1/full/init"
            old=load(old_dir/"scene.json");new=load(dest/"scene.json")
            assert new["input_names"]==old["input_names"]
            assert new["weights_sha256"]==old["weights_sha256"] and new["initial_points"]==old["initial_points"]==100000
            assert new["memory_strategy"]=="cpu_head_features" and new["all_training_views_joint"] and not new["bundle_adjustment"]
            with np.load(old_dir/"scene.npz",allow_pickle=False) as a,np.load(dest/"scene.npz",allow_pickle=False) as b:
                assert np.array_equal(a["names"],b["names"])
                angle=Rotation.from_matrix(np.swapaxes(a["c2w"][:,:3,:3],-1,-2)@b["c2w"][:,:3,:3]).magnitude()*180/np.pi
                extent=float(np.linalg.norm(np.ptp(a["c2w"][:,:3,3],axis=0)))
                center=float(np.max(np.abs(a["c2w"][:,:3,3]-b["c2w"][:,:3,3])))
                focal=float(max(np.max(np.abs(a["K"][:,i,i]-b["K"][:,i,i])/np.abs(a["K"][:,i,i])) for i in [0,1]))
                assert angle.max()<0.02 and center/extent<0.001 and focal<0.001
                cloud_scale=float(np.linalg.norm(np.percentile(a["points"],95,axis=0)-np.percentile(a["points"],5,axis=0)))
                distances=np.concatenate([cKDTree(a["points"]).query(b["points"],workers=1)[0],cKDTree(b["points"]).query(a["points"],workers=1)[0]])
                point_p99=float(np.percentile(distances,99))/cloud_scale
                assert point_p99<0.001
                colors_equal=np.array_equal(a["colors"],b["colors"])
                points_max=float(np.max(np.abs(a["points"]-b["points"]))) if colors_equal else None
            row={"scene":scene,"views":new["registered_frames"],"historical_cpu_disk_seconds":old["initialization_seconds"],
                 "cpu_feature_seconds":new["initialization_seconds"],"observed_time_ratio":old["initialization_seconds"]/new["initialization_seconds"],
                 "historical_peak_gpu_gib":old["peak_gpu_memory_bytes"]/1024**3,"cpu_feature_peak_gpu_gib":new["peak_gpu_memory_bytes"]/1024**3,
                 "max_rotation_difference_degrees":float(angle.max()),"max_center_difference":center,"center_difference_over_extent":center/extent,
                 "max_focal_relative_difference":focal,"point_symmetric_nn_p99_over_cloud_scale":point_p99,"point_parity_passed":True,"initial_point_colors_identical":colors_equal,
                 "max_point_coordinate_difference_if_colors_identical":points_max,"camera_parity_passed":True,"output":str(dest.resolve())}
            state["rows"].append(row);write_json(OUT/"fast_initialization_benchmark.json",state)
            print(json.dumps(row),flush=True)
        state["status"]="completed";write_json(OUT/"fast_initialization_benchmark.json",state)
    except Exception as e:
        state.update(status="failed",error=str(e));write_json(OUT/"fast_initialization_benchmark.json",state);raise
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset");p.add_argument("--output");p.add_argument("--benchmark",action="store_true");a=p.parse_args()
    if a.benchmark:benchmark()
    else:
        if not a.dataset or not a.output:p.error("--dataset and --output are required")
        print(json.dumps(initialize(a.dataset,a.output),indent=2))
