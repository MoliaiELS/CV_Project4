"""GPU parity gate for memory scheduling; never changes the formal test set."""
import gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path("third_party/vggt").resolve()))
from p4.common import write_json
from p4.initialize import train_frames,_repo_info
from p4.vggt_offload import forward,global_block_cpu
from vggt.models.vggt import VGGT
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

root=Path("runs/offload_parity_01"); root.mkdir(exist_ok=False)
torch.manual_seed(42)
with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
    block=Block(64,4,qk_norm=True,rope=RotaryPositionEmbedding2D(),init_values=1.0).eval().cuda()
    x=torch.randn(1,137,64); pos=torch.randint(0,37,(1,137,2))
    native=block(x.cuda(),pos.cuda()).cpu()
    streamed=global_block_cpu(block,x,pos,query_chunk=17)
    torch.testing.assert_close(native,streamed,atol=2e-3,rtol=2e-3)
    block_error=float((native-streamed).abs().max())
del block,native,streamed,x,pos
gc.collect(); torch.cuda.empty_cache()
_repo_info("third_party/vggt","vggt")
_,frames=train_frames("data/part2_sparse/Re10k-1")
paths=[str(Path("data/part2_sparse/Re10k-1")/f["image"]) for f in frames[:2]]
images,_=load_and_preprocess_images_square(paths,target_size=518)
model=VGGT()
weights=torch.load("weights/vggt/model.pt",map_location="cpu",weights_only=True,mmap=True)
model.load_state_dict(weights,strict=True); del weights
model.point_head=model.track_head=None; model.eval()
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
    model.cuda(); pred=model(images.cuda())
    reference={k:pred[k].cpu() for k in ["pose_enc","depth","depth_conf"]}
    del pred
native_peak=torch.cuda.max_memory_allocated()/1024**3
model.cpu(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
    pred=forward(model,images,root/"cache",query_chunk=512,frame_chunk=1)
actual={k:pred[k].cpu() for k in reference}
offload_peak=torch.cuda.max_memory_allocated()/1024**3
del pred,model
gc.collect(); torch.cuda.empty_cache()
with torch.inference_mode():
    re,rk=pose_encoding_to_extri_intri(reference["pose_enc"],(518,518))
    ae,ak=pose_encoding_to_extri_intri(actual["pose_enc"],(518,518))
Rdelta=re[0,:,:,:3].numpy()@np.swapaxes(ae[0,:,:,:3].numpy(),1,2)
angles=np.degrees(Rotation.from_matrix(Rdelta).magnitude())
rel=(actual["depth"]-reference["depth"]).abs()/reference["depth"].abs().clamp_min(1e-6)
report={"status":"pending","frames":2,"query_chunks_exercised":True,"block_max_abs_error":block_error,
        "native_peak_gpu_gib":native_peak,"offload_peak_gpu_gib":offload_peak,
        "max_camera_rotation_difference_degrees":float(angles.max()),
        "max_focal_relative_difference":float(((ak-rk).abs()/(rk.abs().clamp_min(1)))[:,:,[0,1],[0,1]].max()),
        "max_translation_absolute_difference":float((ae[...,-1]-re[...,-1]).abs().max()),
        "depth_median_relative_difference":float(rel.median()),
        "depth_p99_relative_difference":float(torch.quantile(rel.flatten(),.99)),
        "tensor_max_abs":{k:float((actual[k]-reference[k]).abs().max()) for k in reference}}
write_json(root/"parity.json",report); print(json.dumps(report,indent=2),flush=True)
assert report["max_camera_rotation_difference_degrees"]<0.05
assert report["max_focal_relative_difference"]<0.001
assert report["max_translation_absolute_difference"]<0.001
assert report["depth_median_relative_difference"]<0.001 and report["depth_p99_relative_difference"]<0.005
assert all(torch.isfinite(x).all() for x in actual.values())
report["status"]="passed"; write_json(root/"parity.json",report)
print("OFFLOAD PARITY PASSED",flush=True)
