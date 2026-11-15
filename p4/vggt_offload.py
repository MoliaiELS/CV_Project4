"""Memory scheduling for pinned VGGT inference, preserving full-context attention.

Weights, activations and residuals keep the native dtypes. Every global query
attends to EVERY input token. Query batching changes scheduling, never the view
set or the attention graph. Cached DPT features are losslessly stored as float32.
"""
from pathlib import Path
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

def global_block_cpu(block, x, pos, query_chunk=4096):
    """Evaluate an eval-mode global Block with CPU residuals and complete GPU KV."""
    assert not block.training and x.device.type == "cpu" and x.shape[0] == 1
    assert block.attn.fused_attn and block.attn.attn_drop.p == 0
    block.to("cuda")
    _, n, c = x.shape
    h, d = block.attn.num_heads, block.attn.head_dim
    q_cpu = keys = values = None
    for start in range(0, n, query_chunk):
        end = min(start + query_chunk, n)
        y = block.norm1(x[:, start:end].to("cuda"))
        qkv = block.attn.qkv(y).reshape(1, end-start, 3, h, d).permute(2, 0, 3, 1, 4)
        q,k,v = qkv.unbind(0)
        q,k = block.attn.q_norm(q), block.attn.k_norm(k)
        if block.attn.rope is not None:
            p = pos[:, start:end].to("cuda")
            q,k = block.attn.rope(q,p), block.attn.rope(k,p)
        if q_cpu is None:
            q_cpu = torch.empty((1,h,n,d),dtype=q.dtype,device="cpu")
            keys = torch.empty((1,h,n,d),dtype=k.dtype,device="cuda")
            values = torch.empty((1,h,n,d),dtype=v.dtype,device="cuda")
        q_cpu[:,:,start:end].copy_(q.cpu())
        keys[:,:,start:end].copy_(k); values[:,:,start:end].copy_(v)
    del qkv,q,k,v,y
    output = torch.empty_like(x,device="cpu")
    for start in range(0, n, query_chunk):
        end = min(start+query_chunk,n)
        q = q_cpu[:,:,start:end].to("cuda")
        # Complete keys/values: no temporal windows, truncation or token sampling.
        attended = F.scaled_dot_product_attention(q,keys,values,dropout_p=0.0)
        attended = attended.transpose(1,2).reshape(1,end-start,c)
        residual = block.ls1(block.attn.proj_drop(block.attn.proj(attended)))
        z = x[:,start:end].to("cuda") + residual
        z = z + block.ls2(block.mlp(block.norm2(z)))
        output[:,start:end].copy_(z.cpu())
    del keys,values,q_cpu,q,attended,residual,z
    block.to("cpu")
    torch.cuda.empty_cache()
    return output

@torch.inference_mode()
def forward(model, images, cache_dir, query_chunk=4096, frame_chunk=8):
    """CPU model + CPU Sx3xHxW RGB -> camera/depth predictions with native math."""
    assert not model.training and images.device.type == "cpu"
    assert model.point_head is None and model.track_head is None
    a = model.aggregator
    assert a.aa_block_size == 1 and a.aa_order == ["frame","global"]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=False)
    s,_,height,width = images.shape
    t0 = time.monotonic()
    a.patch_embed.to("cuda")
    patches = None
    for start in range(0,s,frame_chunk):
        end = min(start+frame_chunk,s)
        rgb = (images[start:end]-a._resnet_mean[0])/a._resnet_std[0]
        p = a.patch_embed(rgb.to("cuda"))
        if isinstance(p,dict): p=p["x_norm_patchtokens"]
        if patches is None: patches=torch.empty((s,p.shape[1],p.shape[2]),dtype=p.dtype,device="cpu")
        patches[start:end].copy_(p.cpu())
    del p,rgb
    a.patch_embed.to("cpu")
    from vggt.models.aggregator import slice_expand_and_flatten
    tokens = torch.cat([slice_expand_and_flatten(a.camera_token,1,s),
                        slice_expand_and_flatten(a.register_token,1,s),patches],dim=1)
    del patches
    _,patches_per_frame,channels=tokens.shape
    grid=torch.cartesian_prod(torch.arange(height//a.patch_size),torch.arange(width//a.patch_size))
    pos=torch.cat([torch.zeros((s,a.patch_start_idx,2),dtype=grid.dtype),
                   grid[None].expand(s,-1,-1)+1],dim=1)
    files={}
    print(json.dumps({"offload_stage":"patches","frames":s,"tokens_per_frame":patches_per_frame,
                      "seconds":time.monotonic()-t0}),flush=True)
    for layer,(frame,glob) in enumerate(zip(a.frame_blocks,a.global_blocks)):
        tick=time.monotonic()
        frame.to("cuda")
        frame_out=torch.empty_like(tokens)
        for start in range(0,s,frame_chunk):
            end=min(start+frame_chunk,s)
            out=frame(tokens[start:end].to("cuda"),pos=pos[start:end].to("cuda"))
            frame_out[start:end].copy_(out.cpu())
        del out,tokens
        frame.to("cpu"); torch.cuda.empty_cache()
        result=global_block_cpu(glob,frame_out.reshape(1,-1,channels),pos.reshape(1,-1,2),query_chunk)
        tokens=result.reshape(s,patches_per_frame,channels)
        if layer in a.cached_layer_indices:
            path=cache_dir/f"layer_{layer:02d}.npy"
            mapped=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,
                                            shape=(1,s,patches_per_frame,channels*2))
            mapped[0,:,:,:channels]=frame_out.numpy()
            mapped[0,:,:,channels:]=tokens.numpy()
            mapped.flush(); del mapped
            files[layer]=path
        del frame_out,result
        print(json.dumps({"offload_stage":"attention","layer":layer,"frames":s,
                          "global_key_tokens":s*patches_per_frame,"seconds":time.monotonic()-tick,
                          "peak_cuda_gib":torch.cuda.max_memory_allocated()/1024**3}),flush=True)
    del tokens,pos
    # Camera head and DPT retain the official FP32 head execution and frame chunks.
    with torch.autocast(device_type="cuda",enabled=False):
        last=np.load(files[a.depth-1],mmap_mode="r")
        camera_tokens=torch.from_numpy(np.array(last[:,:,0:1],copy=True)).to("cuda")
        del last
        model.camera_head.to("cuda")
        pose=model.camera_head([camera_tokens])[-1]
        del camera_tokens
        model.camera_head.to("cpu")
        model.depth_head.to("cuda")
        depths=[]; confidences=[]
        for start in range(0,s,frame_chunk):
            end=min(start+frame_chunk,s)
            features=[None]*a.depth
            for layer,path in files.items():
                mapped=np.load(path,mmap_mode="r")
                features[layer]=torch.from_numpy(np.array(mapped[:,start:end],copy=True)).to("cuda")
                del mapped
            depth,confidence=model.depth_head._forward_impl(features,images[None,start:end],a.patch_start_idx)
            depths.append(depth.cpu()); confidences.append(confidence.cpu())
            del features,depth,confidence
        model.depth_head.to("cpu")
    torch.cuda.empty_cache()
    return {"pose_enc":pose,"depth":torch.cat(depths,dim=1),"depth_conf":torch.cat(confidences,dim=1)}
