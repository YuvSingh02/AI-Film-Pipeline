#!/usr/bin/env python3
"""
AI video pipeline - everything tested so far, in one file.

Run on a fresh Vast.ai instance (template: vastai/comfy, RTX 5000 Ada 32GB
or similar, 150GB+ disk). Upload this file AND video_ltx2_5_i2v.json
(the ComfyUI API-format LTX-2.5 image-to-video workflow) to /workspace.

Commands (run with the same `python` you used before):
    python film_pipeline.py setup            # install libs + download all models
    python film_pipeline.py setup --with-wan # also download Wan 2.2 (backup model, ~37GB)
    python film_pipeline.py check            # verify GPU, disk, models, workflow, ComfyUI
    python film_pipeline.py inspect          # list nodes of the workflow file
    python film_pipeline.py keyframes        # 4 test keyframes (Krea-2 + Realism LoRA)
    python film_pipeline.py keyframes --prompts shots.json --prefix kf
    python film_pipeline.py animate kf_0.png "Slow camera push..." --seconds 5
    python film_pipeline.py voice "Narration text" --voice am_adam
    python film_pipeline.py stock "private jet on airport apron" stock_test
    python film_pipeline.py free             # make ComfyUI release GPU memory
"""
import argparse
import getpass
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

WS = "/workspace"
COMFY = f"{WS}/ComfyUI"
MODELS = f"{COMFY}/models"
INPUT = f"{COMFY}/input"
OUTPUT = f"{COMFY}/output"
KREA_DIR = f"{WS}/models/krea2-turbo"
KREA_LORA_DIR = f"{WS}/models/krea2-realism-lora"
WORKFLOW = f"{WS}/video_ltx2_5_i2v.json"
TTS_ENV = f"{WS}/tts_env"
CREDITS = f"{WS}/credits.jsonl"
TMP = f"{WS}/tmp_dl"

# ---- model files (all verified during the first session) -----------------
LTX_REPO = "Lightricks/LTX-2.5"  # gated: accept license on Hugging Face first
LTX_FILES = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
]
# the workflow also loads this (prompt-enhancer text encoder), public repo
GEMMA_E2B = ("Comfy-Org/gemma-4", "text_encoders/gemma4_e2b_it_int8_convrot.safetensors")

W22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
W21 = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
WAN_FILES = [  # (repo, path in repo, destination folder under ComfyUI/models)
    (W22, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
    (W22, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "diffusion_models"),
    (W21, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    (W21, "split_files/vae/wan_2.1_vae.safetensors", "vae"),
    (W22, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "loras"),
    (W22, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "loras"),
]

# ---- node ids inside video_ltx2_5_i2v.json (from the first session) ------
NODES = {
    "image": ("395", "image"),
    "prompt": ("398:376", "value"),
    "seconds": ("398:362", "value"),
    "seed_a": ("398:339", "noise_seed"),
    "seed_b": ("398:338", "noise_seed"),
    "save": ("75", "filename_prefix"),
}

TEST_PROMPTS = [
    "A photorealistic wide shot of a private jet parked on a tarmac at sunrise, long shadows, cold teal haze, cinematic documentary",
    "A photorealistic wide shot of an empty airport runway at night with blue edge lights, thin fog, quiet suspense, cinematic documentary",
    "A photorealistic wide shot of an old aviation flight log book with handwritten entries on a dark wooden desk, low key lighting, dust in the air",
    "A photorealistic wide shot of a misty ocean at dawn, rolling waves, soft golden light on the water, cinematic documentary",
]


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


# ============================== setup ======================================
def have_diffusers_krea():
    r = subprocess.run([sys.executable, "-c", "from diffusers import Krea2Pipeline"],
                       capture_output=True)
    return r.returncode == 0


def get_token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        tok = getpass.getpass("Hugging Face READ token (hidden, needed for LTX-2.5): ").strip()
        os.environ["HF_TOKEN"] = tok
    return tok


def cmd_setup(args):
    if not os.path.isdir(COMFY):
        sys.exit(f"{COMFY} not found - use the vastai/comfy template.")
    sh("apt-get install -y espeak-ng >/dev/null 2>&1", check=False)
    if not have_diffusers_krea():
        sh(f"{sys.executable} -m pip install -q -U diffusers")
    sh(f"{sys.executable} -m pip install -q -U accelerate peft pillow huggingface_hub")
    if not os.path.exists(f"{TTS_ENV}/bin/python"):
        sh(f"python3 -m venv {TTS_ENV}")
    sh(f"{TTS_ENV}/bin/pip install -q kokoro soundfile")

    from huggingface_hub import hf_hub_download, snapshot_download
    token = get_token()

    for f in LTX_FILES:
        dest = f"{MODELS}/{f}"
        if os.path.exists(dest):
            print("have", os.path.basename(f)); continue
        print("downloading", os.path.basename(f), flush=True)
        hf_hub_download(LTX_REPO, f, local_dir=MODELS, token=token)

    repo, f = GEMMA_E2B
    if os.path.exists(f"{MODELS}/{f}"):
        print("have", os.path.basename(f))
    else:
        print("downloading", os.path.basename(f), flush=True)
        hf_hub_download(repo, f, local_dir=MODELS, token=token)

    for repo, d in (("krea/Krea-2-Turbo", KREA_DIR), ("gokaygokay/Krea-2-Realism-LoRA", KREA_LORA_DIR)):
        print("downloading", repo, flush=True)
        snapshot_download(repo_id=repo, local_dir=d, token=token)

    if args.with_wan:
        for repo, f, dest in WAN_FILES:
            target = f"{MODELS}/{dest}/{os.path.basename(f)}"
            if os.path.exists(target):
                print("have", os.path.basename(f)); continue
            print("downloading", os.path.basename(f), flush=True)
            p = hf_hub_download(repo, f, local_dir=TMP, token=token)
            shutil.move(p, target)

    shutil.rmtree(TMP, ignore_errors=True)
    print("SETUP DONE. Next: python film_pipeline.py check")


# ============================== check / inspect ============================
def comfy_port():
    for p in (8188, 18188):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{p}/system_stats", timeout=3)
            return p
        except Exception:
            pass
    return None


def cmd_check(_):
    sh("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv", check=False)
    sh(f"df -h {WS} | tail -1", check=False)
    missing = 0
    files = [f"{MODELS}/{f}" for f in LTX_FILES] + [f"{MODELS}/{GEMMA_E2B[1]}"]
    for p in files + [KREA_DIR + "/model_index.json", KREA_LORA_DIR]:
        ok = os.path.exists(p)
        missing += not ok
        print("OK     " if ok else "MISSING", p)
    wf = os.path.exists(WORKFLOW)
    print("OK     " if wf else "MISSING", WORKFLOW,
          "" if wf else "  <- upload it (or re-export: ComfyUI > Workflow > Export (API))")
    if wf:
        w = json.load(open(WORKFLOW))
        bad = [k for k, (n, _f) in NODES.items() if n not in w]
        print("workflow nodes:", "OK" if not bad else f"IDs changed for {bad}, run: inspect")
    print("kokoro env:", "OK" if os.path.exists(f"{TTS_ENV}/bin/python") else "MISSING")
    port = comfy_port()
    print("ComfyUI:", f"running on port {port}" if port else "NOT reachable")
    print("ALL GOOD" if not missing and wf and port else "Fix the items above first.")


def cmd_inspect(_):
    w = json.load(open(WORKFLOW))
    for k, n in w.items():
        ins = {a: (b if isinstance(b, (int, float, bool)) or (isinstance(b, str) and len(b) < 70) else "...")
               for a, b in n["inputs"].items() if not isinstance(b, list)}
        print(k, n["class_type"], ins)


# ============================== free GPU memory ============================
def free_comfy():
    for p in (8188, 18188):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{p}/free",
                json.dumps({"unload_models": True, "free_memory": True}).encode(),
                {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            print(f"ComfyUI memory freed (port {p})")
        except Exception:
            pass
    time.sleep(3)


def cmd_free(_):
    free_comfy()
    sh("nvidia-smi --query-gpu=memory.used --format=csv", check=False)


# ============================== keyframes (Krea-2) =========================
def cmd_keyframes(args):
    import torch
    from diffusers import Krea2Pipeline
    prompts = TEST_PROMPTS
    if args.prompts:
        raw = json.load(open(args.prompts))
        prompts = [x if isinstance(x, str) else x["image_prompt"] for x in raw]
    free_comfy()  # ComfyUI keeps video models in VRAM; Krea-2 needs the space
    pipe = Krea2Pipeline.from_pretrained(KREA_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(KREA_LORA_DIR)
    os.makedirs(INPUT, exist_ok=True)
    for i, p in enumerate(prompts):
        g = torch.Generator(device="cpu").manual_seed(args.seed + i)
        img = pipe(p, num_inference_steps=14, guidance_scale=1.8, width=1536, height=832,
                   attention_kwargs={"scale": 1.0}, generator=g).images[0]
        out = f"{INPUT}/{args.prefix}_{i}.png"
        img.save(out)
        print("saved", out, flush=True)
    del pipe
    torch.cuda.empty_cache()
    print("KEYFRAMES DONE")


# ============================== animate (LTX-2.5 via ComfyUI) ==============
def animate(image, prompt, seconds=5, seed=None, prefix="video/shot", timeout=1800):
    w = json.load(open(WORKFLOW))
    for key, (node, _f) in NODES.items():
        if node not in w:
            sys.exit(f"Workflow node {node} ({key}) not found - run: python film_pipeline.py inspect")
    seed = random.randint(1, 2**31) if seed is None else seed
    vals = {"image": image, "prompt": prompt, "seconds": seconds,
            "seed_a": seed, "seed_b": seed, "save": prefix}
    for key, v in vals.items():
        node, field = NODES[key]
        w[node]["inputs"][field] = v
    port = comfy_port()
    if not port:
        sys.exit("ComfyUI not reachable")
    base = f"http://127.0.0.1:{port}"
    req = urllib.request.Request(base + "/prompt", json.dumps({"prompt": w}).encode(),
                                 {"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    print("queued", pid, flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.load(urllib.request.urlopen(f"{base}/history/{pid}"))
        if pid in h:
            h = h[pid]
            if h["status"]["status_str"] == "error":
                sys.exit("FAILED: " + str(h["status"]["messages"])[-600:])
            return [f"{OUTPUT}/{i.get('subfolder', '')}/{i['filename']}"
                    for o in h["outputs"].values() for v in o.values() if isinstance(v, list)
                    for i in v if isinstance(i, dict) and "filename" in i]
        time.sleep(3)
    sys.exit("Timed out waiting for ComfyUI")


def cmd_animate(args):
    t = time.time()
    print(animate(args.image, args.prompt, args.seconds, args.seed, args.prefix))
    print(f"took {time.time() - t:.0f}s")


# ============================== voice (Kokoro, own venv) ===================
KOKORO_CODE = r'''
import sys, numpy as np, soundfile as sf
from kokoro import KPipeline
text, voice, out = sys.argv[1], sys.argv[2], sys.argv[3]
pipe = KPipeline(lang_code=voice[0])   # 'a' = US voices (am_*), 'b' = UK voices (bm_*)
audio = np.concatenate([a for _, _, a in pipe(text, voice=voice)])
sf.write(out, audio, 24000)
print("saved", out, round(len(audio) / 24000, 1), "sec")
'''


def cmd_voice(args):
    out = args.out or f"{WS}/voice_{args.voice}.wav"
    subprocess.run([f"{TTS_ENV}/bin/python", "-c", KOKORO_CODE, args.text, args.voice, out], check=True)


# ============================== stock images (Wikimedia Commons) ===========
def _get(url):
    ua = {"User-Agent": "PulseParadoxPipeline/0.1 (personal video project)"}
    return urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=60).read()


def stock_fetch(query, name, min_w=1600):
    """Public-domain / CC0 / plain CC BY only (no -SA, -NC, -ND). Credit is logged."""
    from PIL import Image
    q = urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": query + " filetype:bitmap", "gsrlimit": 25, "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1920, "format": "json"})
    data = json.loads(_get("https://commons.wikimedia.org/w/api.php?" + q))
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    for p in pages:
        ii = p["imageinfo"][0]
        md = ii["extmetadata"]
        lic = md.get("LicenseShortName", {}).get("value", "").lower()
        if not lic.startswith(("cc0", "public domain", "pd", "cc by")):
            continue
        if any(x in lic for x in ("-sa", "-nc", "-nd")):
            continue
        if ii["width"] < min_w or ii["width"] < ii["height"]:
            continue
        img = Image.open(io.BytesIO(_get(ii["thumburl"]))).convert("RGB")
        w, h = img.size
        th = int(w * 9 / 16)
        top = max(0, (h - th) // 2)
        img = img.crop((0, top, w, min(h, top + th))).resize((1536, 864))
        out = f"{INPUT}/{name}.png"
        img.save(out)
        credit = {"file": out, "title": p["title"], "license": md["LicenseShortName"]["value"],
                  "author": re.sub("<[^>]+>", "", md.get("Artist", {}).get("value", "unknown")),
                  "page": ii["descriptionurl"]}
        with open(CREDITS, "a") as f:
            f.write(json.dumps(credit) + "\n")
        return credit
    return None


def cmd_stock(args):
    print(stock_fetch(args.query, args.name))


# ============================== CLI ========================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup"); s.add_argument("--with-wan", action="store_true"); s.set_defaults(fn=cmd_setup)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("inspect").set_defaults(fn=cmd_inspect)
    sub.add_parser("free").set_defaults(fn=cmd_free)

    s = sub.add_parser("keyframes")
    s.add_argument("--prompts", help="JSON list of prompt strings (or objects with image_prompt)")
    s.add_argument("--prefix", default="kf")
    s.add_argument("--seed", type=int, default=100)
    s.set_defaults(fn=cmd_keyframes)

    s = sub.add_parser("animate")
    s.add_argument("image", help="file name inside ComfyUI/input, e.g. kf_0.png")
    s.add_argument("prompt")
    s.add_argument("--seconds", type=int, default=5)
    s.add_argument("--seed", type=int)
    s.add_argument("--prefix", default="video/shot")
    s.set_defaults(fn=cmd_animate)

    s = sub.add_parser("voice")
    s.add_argument("text")
    s.add_argument("--voice", default="am_adam")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_voice)

    s = sub.add_parser("stock")
    s.add_argument("query")
    s.add_argument("name")
    s.set_defaults(fn=cmd_stock)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
