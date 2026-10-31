import glob, json, math, os, shutil, subprocess, sys
sys.path.insert(0, "/workspace")
import film_pipeline as fp

FILM = "/workspace/film"
SHOTS = "/workspace/shots.json"
AUDIO = f"{FILM}/audio"
VOICE = "am_adam"

KOKORO_ALL = r'''
import sys, json, numpy as np, soundfile as sf
from kokoro import KPipeline
shots = json.load(open(sys.argv[1])); voice = sys.argv[2]; outdir = sys.argv[3]
pipe = KPipeline(lang_code=voice[0])
durs = []
for i, s in enumerate(shots):
    audio = np.concatenate([a for _, _, a in pipe(s["narration"], voice=voice)])
    sf.write(f"{outdir}/scene_{i:03d}.wav", audio, 24000)
    durs.append(round(len(audio) / 24000, 2)); print(i, durs[-1], flush=True)
json.dump(durs, open(sys.argv[4], "w"))
print("total", round(sum(durs), 1), "sec")
'''

def load():
    return json.load(open(SHOTS)), json.load(open(f"{FILM}/durations.json"))

def narrate():
    os.makedirs(AUDIO, exist_ok=True)
    subprocess.run([f"{fp.TTS_ENV}/bin/python", "-c", KOKORO_ALL, SHOTS, VOICE, AUDIO, f"{FILM}/durations.json"], check=True)

def animate_all():
    shots, durs = load()
    for i, s in enumerate(shots):
        pre = f"video/scene_{i:03d}"
        if glob.glob(f"{fp.OUTPUT}/{pre}_*.mp4"):
            print("skip", i); continue
        if not os.path.exists(f"{fp.INPUT}/scene_{i}.png"):
            sys.exit(f"missing keyframe scene_{i}.png - run the keyframes step first")
        secs = min(10, max(4, math.ceil(durs[i] + 0.5)))
        fp.animate(f"scene_{i}.png", s["motion_prompt"], secs, None, pre)
        print("clip", i, "done,", secs, "s", flush=True)
    print("ALL CLIPS DONE")

def assemble():
    if not shutil.which("ffmpeg"):
        subprocess.run("apt-get install -y ffmpeg >/dev/null 2>&1", shell=True)
    shots, durs = load()
    os.makedirs(f"{FILM}/scenes", exist_ok=True)
    outs = []
    for i in range(len(shots)):
        clip = sorted(glob.glob(f"{fp.OUTPUT}/video/scene_{i:03d}_*.mp4"))[-1]
        wav = f"{AUDIO}/scene_{i:03d}.wav"
        out = f"{FILM}/scenes/scene_{i:03d}.mp4"
        has_a = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                     "stream=codec_type", "-of", "csv=p=0", clip], capture_output=True, text=True).stdout.strip())
        fc = ("[0:a]volume=0.25[a0];[1:a]apad[a1];[a0][a1]amix=inputs=2:duration=longest:normalize=0[a]"
              if has_a else "[1:a]apad[a]")
        r = subprocess.run(["ffmpeg", "-y", "-i", clip, "-i", wav, "-filter_complex",
                            "[0:v]scale=1280:720:flags=lanczos,fps=24,format=yuv420p[v];" + fc,
                            "-map", "[v]", "-map", "[a]", "-t", f"{durs[i] + 0.5:.2f}", "-c:v", "libx264",
                            "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", out],
                           capture_output=True, text=True)
        if not os.path.exists(out):
            sys.exit(f"scene {i} failed: {r.stderr[-500:]}")
        outs.append(out); print("scene", i, "ok", flush=True)
    lst = f"{FILM}/concat.txt"
    open(lst, "w").write("\n".join(f"file '{o}'" for o in outs) + "\n")
    final = f"{FILM}/FINAL_v1.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final],
                   capture_output=True)
    print("DONE", final, os.path.getsize(final) // 1024 // 1024, "MB")

if __name__ == "__main__":
    {"narrate": narrate, "animate": animate_all, "assemble": assemble}[sys.argv[1]]()
