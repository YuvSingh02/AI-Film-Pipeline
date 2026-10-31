# AI Film Pipeline

An experimental, Python-based pipeline for automating cinematic documentary and storytelling videos using generative AI.

## Overview

This project explores an end-to-end AI filmmaking workflow, combining image generation, image-to-video generation, synthetic narration, AI talking-head animation and automated video assembly.

The goal is to reduce manual production work while maintaining cinematic visual quality.

## Core Technologies

- Python
- ComfyUI API
- LTX-2.5
- Wan 2.2 / InfiniteTalk
- Krea-2 and Realism LoRA
- Kokoro TTS
- FFmpeg
- PyTorch and Diffusers
- Vast.ai GPU infrastructure

## Pipeline Architecture

1. Story and shot planning
2. Narration generation
3. AI keyframe generation
4. Image-to-video generation
5. AI talking-head generation
6. Video processing and assembly

## Project Structure

- src/ — Python pipeline scripts
- workflows/ — ComfyUI workflow exports
- docs/ — Project documentation
- examples/ — Sample generated video
- shots.json — Example scene definitions

## Demonstration

**Flight Logs — AI-generated documentary experiment**

[View the generated video](examples/flight-logs-demo.mp4)

## Infrastructure

Developed and tested using rented GPU infrastructure on Vast.ai.

Test environment:
- NVIDIA RTX 5000 Ada
- 32 GB GPU VRAM
- ComfyUI
- Python and PyTorch

## Current Limitations

- Independent scene generation can produce visual continuity inconsistencies.
- Talking-head generation requires video-extension passes for longer narration.
- Facial identity consistency needs further improvement.
- Automated quality assessment and final editing are still under development.

## Roadmap

- [ ] Improve scene-to-scene visual continuity
- [ ] Implement talking-head video extension
- [ ] Integrate audio and sound design
- [ ] Add automated video quality checks
- [ ] Implement captions and final color grading
- [ ] Build a unified host, AI B-roll and stock-footage timeline

## Model Access and Licensing

This repository does not redistribute model weights.

Each model has separate access requirements and licensing terms. Check the original model repositories before downloading or using them commercially.

## Responsible AI

- Use only authorized faces and voices.
- Disclose realistic AI-generated content.
- Respect third-party model and stock-media licenses.

---

**Project:** Learning AI Film Making Using Open Source Models
**Focus:** Generative AI | Multimodal AI | Video Automation | Python
