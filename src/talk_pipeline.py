#!/usr/bin/env python3
"""
Talking-head pipeline (InfiniteTalk, single speaker, photo + audio -> talking video).
Separate from film_pipeline.py (B-roll pipeline); we combine them later.

Fresh Vast.ai instance (template: vastai/comfy, 32GB GPU, ~100GB disk is enough):
  1. Upload this file and your host photo (e.g. host.jpg) to /workspace
  2. python talk_pipeline.py setup         # downloads ~30GB of models (no token needed)
  3. python talk_pipeline.py check
  4. python talk_pipeline.py talk host.jpg my_audio.wav "A man talks to the camera, natural head and hand movement."

Notes (from the first test):
  * Audio is auto-trimmed to ~3.2 s: one pass = 81 frames at 25 fps.
    Longer narration needs the "video extend" step, which is NOT built or tested yet.
  * Test result: single photo works (93 s for 3 s of video on an RTX 5000 Ada) but the
    face drifted a little. Next test: video-to-video from a short clip of the host.
  * Defaults are portrait 480x832 (matches a phone selfie). For landscape use --width 832 --height 480.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

WS = "/workspace"
COMFY = f"{WS}/ComfyUI"
MODELS = f"{COMFY}/models"
INPUT = f"{COMFY}/input"
TMP = f"{WS}/tmp_dl"

KJ = "Kijai/WanVideo_comfy_fp8_scaled"
KJ2 = "Kijai/WanVideo_comfy"
W21 = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
FILES = [  # (repo, path in repo, folder under ComfyUI/models)
    (KJ, "I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models"),
    (W21, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    (W21, "split_files/model_patches/wan2.1_infiniteTalk_single_fp16.safetensors", "model_patches"),
    ("Kijai/wav2vec2_safetensors", "wav2vec2-chinese-base_fp16.safetensors", "audio_encoders"),
    (KJ2, "Wan2_1_VAE_bf16.safetensors", "vae"),
    (KJ2, "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras"),
]


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def comfy_port():
    for p in (8188, 18188):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{p}/system_stats", timeout=3)
            return p
        except Exception:
            pass
    return None


# ------------------------------- setup / check -----------------------------
def cmd_setup(_):
    if not os.path.isdir(COMFY):
        sys.exit(f"{COMFY} not found - use the vastai/comfy template.")
    sh(f"{sys.executable} -m pip install -q -U huggingface_hub pillow")
    from huggingface_hub import hf_hub_download
    for repo, f, dest in FILES:
        target = f"{MODELS}/{dest}/{os.path.basename(f)}"
        if os.path.exists(target):
            print("have", os.path.basename(f)); continue
        print("downloading", os.path.basename(f), flush=True)
        p = hf_hub_download(repo, f, local_dir=TMP)
        os.makedirs(f"{MODELS}/{dest}", exist_ok=True)
        shutil.move(p, target)
    shutil.rmtree(TMP, ignore_errors=True)
    print("SETUP DONE. Next: python talk_pipeline.py check")


def cmd_check(_):
    sh("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv", check=False)
    sh(f"df -h {WS} | tail -1", check=False)
    missing = 0
    for repo, f, dest in FILES:
        p = f"{MODELS}/{dest}/{os.path.basename(f)}"
        ok = os.path.exists(p)
        missing += not ok
        print("OK     " if ok else "MISSING", p)
    port = comfy_port()
    print("ComfyUI:", f"running on port {port}" if port else "NOT reachable")
    print("ALL GOOD" if not missing and port else "Fix the items above first.")


# ------------------------------- talking head ------------------------------
def build(image, audio, prompt, w, h, length, seed, prefix):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "Wan2_1_VAE_bf16.safetensors"}},
        "6": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": "wav2vec2-chinese-base_fp16.safetensors"}},
        "7": {"class_type": "ModelPatchLoader", "inputs": {"name": "wan2.1_infiniteTalk_single_fp16.safetensors"}},
        "8": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["3", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["9", 0]}},
        "11": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["6", 0], "audio": ["2", 0]}},
        "12": {"class_type": "WanInfiniteTalkToVideo", "inputs": {
            "mode": "single_speaker", "model": ["8", 0], "model_patch": ["7", 0],
            "positive": ["9", 0], "negative": ["10", 0], "vae": ["5", 0],
            "width": w, "height": h, "length": length, "start_image": ["1", 0],
            "audio_encoder_output_1": ["11", 0], "motion_frame_count": 9, "audio_scale": 1.0}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "CFGGuider", "inputs": {"model": ["12", 0], "positive": ["12", 1], "negative": ["12", 2], "cfg": 1.0}},
        "15": {"class_type": "BasicScheduler", "inputs": {"model": ["12", 0], "scheduler": "normal", "steps": 6, "denoise": 1.0}},
        "16": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "17": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["16", 0], "guider": ["14", 0], "sampler": ["13", 0], "sigmas": ["15", 0], "latent_image": ["12", 3]}},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["5", 0]}},
        "19": {"class_type": "CreateVideo", "inputs": {"images": ["18", 0], "audio": ["2", 0], "fps": 25}},
        "20": {"class_type": "SaveVideo", "inputs": {"video": ["19", 0], "filename_prefix": prefix, "format": "auto", "format.codec": "auto", "codec": "auto"}},
    }


def to_input(path):
    """Copy a file into ComfyUI/input (if given as a path) and return its file name."""
    if os.path.exists(path):
        name = os.path.basename(path).replace(" ", "_")
        dst = f"{INPUT}/{name}"
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copy(path, dst)
        return name
    name = os.path.basename(path)
    if os.path.exists(f"{INPUT}/{name}"):
        return name
    sys.exit(f"File not found: {path}")


def cmd_talk(args):
    image = to_input(args.image if os.path.exists(args.image) else f"{WS}/{args.image}")
    audio_src = args.audio if os.path.exists(args.audio) else f"{WS}/{args.audio}"
    audio_name = to_input(audio_src)
    secs = (args.length - 1) / 25
    trimmed = f"talk_in_{int(time.time())}.wav"
    subprocess.run(["ffmpeg", "-y", "-i", f"{INPUT}/{audio_name}", "-t", f"{secs:.2f}", f"{INPUT}/{trimmed}"],
                   capture_output=True)
    if not os.path.exists(f"{INPUT}/{trimmed}"):
        sys.exit("ffmpeg could not read the audio (is ffmpeg installed?)")
    seed = args.seed if args.seed is not None else random.randint(1, 2**31)
    base = f"http://127.0.0.1:{comfy_port() or sys.exit('ComfyUI not reachable')}"
    graph = build(image, trimmed, args.prompt, args.width, args.height, args.length, seed, args.prefix)
    req = urllib.request.Request(base + "/prompt", json.dumps({"prompt": graph}).encode(), {"Content-Type": "application/json"})
    t = time.time()
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit("REJECTED: " + e.read().decode()[:1500])
    print("queued", pid, flush=True)
    while True:
        h = json.load(urllib.request.urlopen(f"{base}/history/{pid}"))
        if pid in h:
            h = h[pid]
            if h["status"]["status_str"] == "error":
                sys.exit("FAILED: " + str(h["status"]["messages"])[-800:])
            break
        time.sleep(3)
    files = [f"{COMFY}/output/{i.get('subfolder', '')}/{i['filename']}"
             for o in h["outputs"].values() for v in o.values() if isinstance(v, list)
             for i in v if isinstance(i, dict) and "filename" in i]
    print(files, f"took {time.time() - t:.0f}s")
    from PIL import Image, ImageStat
    chk = f"{INPUT}/talk_check.png"
    subprocess.run(["ffmpeg", "-y", "-ss", "1.5", "-i", files[0], "-frames:v", "1", chk], capture_output=True)
    print("frame brightness:", round(ImageStat.Stat(Image.open(chk).convert("L")).mean[0], 1), "(0 = black)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    s = sub.add_parser("talk")
    s.add_argument("image", help="host photo, e.g. host.jpg (in /workspace or ComfyUI/input)")
    s.add_argument("audio", help="narration audio (wav/mp3)")
    s.add_argument("prompt", nargs="?", default="A man talks to the camera, natural head and hand movement.")
    s.add_argument("--width", type=int, default=480)
    s.add_argument("--height", type=int, default=832)
    s.add_argument("--length", type=int, default=81, help="frames at 25 fps (81 = 3.2 s)")
    s.add_argument("--seed", type=int)
    s.add_argument("--prefix", default="video/talk")
    s.set_defaults(fn=cmd_talk)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
