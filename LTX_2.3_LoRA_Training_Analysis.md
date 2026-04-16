# LTX-2.3 LoRA Training: Architecture & Training Guide

*Based on direct codebase inspection of `/Users/efrattaig/projects/sm/LTX-2`. All class names and parameter values are verified against the source.*

---

## What Is LoRA and Why Are We Using It?

The LTX-2.3 model has **22 billion parameters** — the numbers that define everything it knows about how video and audio look and sound. Retraining all of them from scratch on Skye footage would take weeks, cost thousands of dollars in compute, and likely destroy the general video-generation ability the model already has.

**LoRA (Low-Rank Adaptation)** solves this by adding a tiny parallel layer next to certain existing layers in the model — instead of changing the original layer, you train a small "correction" on top of it. Concretely:

- The original weight matrix stays **completely frozen** (unchanged)
- Next to it, two small matrices are inserted: `A` (input compressor) and `B` (output expander)
- During training, only `A` and `B` learn. During inference: `output = original(x) + B(A(x)) * scale`
- The rank parameter (e.g. rank=32) is the "bottleneck size" of `A` and `B` — lower rank = fewer parameters = faster but less capacity

The result is a small `.safetensors` file (~50–200 MB) that teaches the model new concepts — in our case, Skye's appearance, voice, and movement — without modifying the base 22B weights at all.

---

## What Parts of the Model Do We Target?

### The Overall Structure

LTX-2.3 is a **Diffusion Transformer** — it generates video by starting from pure noise and gradually denoising it, step by step, guided by a text prompt. The "brain" of this process is a stack of **48 identical processing blocks**, each of which asks: *given what I know so far, how should I refine the video and audio at this denoising step?*

Each block handles both video and audio simultaneously and contains several **attention modules** plus a **feed-forward network**:

```
LTXModel
└── 48 transformer blocks, each containing:
    ├── VIDEO BRANCH
    │   ├── attn1  — video self-attention
    │   ├── attn2  — video attends to text prompt
    │   └── ff     — video feed-forward network
    ├── AUDIO BRANCH
    │   ├── audio_attn1  — audio self-attention
    │   ├── audio_attn2  — audio attends to text prompt
    │   └── audio_ff     — audio feed-forward network
    └── AUDIO↔VIDEO CROSS-ATTENTION
        ├── audio_to_video_attn  — video attends to audio
        └── video_to_audio_attn  — audio attends to video
```

### What Is Attention?

Attention is how the model figures out which parts of the input should influence each other. You can think of it as a "routing" system: for each position in the video (a pixel region at a given time), attention decides which other positions are most relevant and how much to draw from them.

Each attention module does this with three learnable projections:
- **Q (Query)** — "what am I looking for?"
- **K (Key)** — "what does each other position offer?"
- **V (Value)** — "what information does each position actually contain?"

The match between Q and K determines how much attention is paid; then V is retrieved accordingly. These are implemented as simple linear layers (`to_q`, `to_k`, `to_v`, `to_out`), which is exactly why they're good LoRA targets — they're the decision-making weights.

**The six attention types in each block:**

| Module | What it does |
|--------|-------------|
| `attn1` (video self-attn) | Video frames attend to each other — spatial and temporal consistency |
| `attn2` (video cross-attn) | Video attends to the text prompt — "does this frame match the description?" |
| `audio_attn1` (audio self-attn) | Audio segments attend to each other — temporal audio coherence |
| `audio_attn2` (audio cross-attn) | Audio attends to the text prompt — "does this sound match the description?" |
| `audio_to_video_attn` | Video frames attend to audio — "what does the audio say the video should look like?" |
| `video_to_audio_attn` | Audio attends to video — "what should the audio sound like given what's on screen?" |

### What We Actually Train

We apply LoRA to the `to_q`, `to_k`, `to_v`, and `to_out.0` linear layers **inside every attention module** across all 48 blocks. That covers all six attention types above — 28 weight matrices per block × 48 blocks = 1,344 total layers modified.

**Why these four and not others?** Because they are `nn.Linear` layers (simple matrix multiplications), which is exactly what LoRA's `B @ A` structure can augment. The normalization layers next to them (`q_norm`, `k_norm`) use a different mathematical structure (RMSNorm) that doesn't have a weight matrix LoRA can meaningfully wrap.

In the config, you just write:
```yaml
lora:
  target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
```
The training framework (PEFT) automatically finds every layer in the model whose name ends with these strings and injects LoRA into all of them.

### Feed-Forward Layers (Optional)

Each block also has a **feed-forward network (FFN)** — two linear layers with a nonlinearity between them. The FFN processes each position independently after attention, acting like a "memory" that stores associations. For video, it's `4096 → 16384 → 4096` dimensions.

You can also add LoRA to the FFN (`ff.net.0.proj`, `ff.net.2`). This increases the model's capacity to learn the new concept but also increases training time and the risk of overfitting.

**Rule of thumb:**
- Attention-only LoRA → good for character appearance, voice consistency
- Attention + FFN LoRA → needed only if the character's *motion style* or *behavior* isn't being captured

---

## Why LTX-2.3 (22B) and Not the Older 19B Model

The 22B model has a feature called **`cross_attention_adaln`** that the 19B model does not. Here's what this means:

In attention modules, the model also conditions on **how noisy the current video is** (a signal called `sigma` — high at the start of denoising, near zero at the end). In LTX-2.3, this noise level is used to dynamically scale and shift the attention computation itself at every block — a technique called **AdaLN (Adaptive Layer Normalization)**. It's like the model automatically adjusting its "confidence" in the text prompt based on how far along in the denoising process it is.

The 19B model doesn't do this — its cross-attention is noise-level-agnostic. The 22B model's improved prompt-following behavior (more faithful to what you described in text) comes directly from this mechanism.

**Why this matters for LoRA:** A LoRA trained on the 19B model cannot be transferred to the 22B model because the weight structures are different. And if you're using 22B for inference, you must train on 22B. The correct checkpoint is `ltx-2.3-22b-dev.safetensors`.

Other 22B improvements over 19B:
| Component | 19B | 22B |
|-----------|-----|-----|
| Text conditioning per block | Fixed | Dynamically scaled by noise level |
| Text → video/audio routing | Single stream for both | Separate pathways for video vs. audio |
| Audio quality | HiFi-GAN vocoder | BigVGAN v2 + bandwidth extension (better quality, wider freq range) |

Version detection is automatic — the trainer reads the model's internal config and picks the right code path without any manual switches.

---

## Training Parameters Explained

### Rank and Alpha

**Rank** is the bottleneck size of the LoRA matrices. Higher rank = more parameters = more capacity to learn, but also slower training and higher overfitting risk.

- Rank 16 → ~50M trainable parameters — good for a fast validation run
- Rank 32 → ~100M trainable parameters — standard for character LoRA
- Rank 64 → ~200M parameters — only if rank 32 is clearly underfitting

**Alpha** controls how strongly the LoRA correction is applied. The scale factor is `alpha / rank`. With `alpha = rank` (our setup), this equals 1.0 — meaning the LoRA correction is added at full strength. Increasing alpha relative to rank amplifies the LoRA; decreasing it dampens it. Start with `alpha = rank` and don't touch it unless results are too weak or overfit.

### Learning Rate

`1e-4` is the standard for LoRA training. This means the weights move 0.0001 of the gradient direction per step — small enough to not destroy what the model already knows, large enough to make meaningful progress.

### Steps

With 409 clips, the key question is how many times the model sees each clip. At batch_size=1:
- 1000 steps → each clip seen ~2.4 times on average
- 2500 steps → each clip seen ~6 times
- 5000 steps → ~12 times

More steps = better character memorization, but also higher overfitting risk if the concept is narrow. The validation videos at each checkpoint are how you know when to stop.

### Precision: Always BF16

**BF16 (bfloat16)** is a 16-bit number format designed for deep learning. It has a wider range than FP16 (regular 16-bit) but lower decimal precision. For a 22B model:

- **FP32** (full 32-bit): 2× the VRAM of BF16, no meaningful quality gain for this use case
- **BF16**: The right choice — what the model was originally trained in
- **FP16**: Do not use — the model produces NaN (not-a-number) errors at this scale due to numerical overflow

**Gradient checkpointing** (`enable_gradient_checkpointing: true`) trades compute for memory: instead of keeping all intermediate activations in GPU memory during the backward pass, it recomputes them on the fly. This cuts VRAM by ~40% at the cost of ~20% slower training. Always enable for a 22B model.

### Timestep Sampling

During training, each video is artificially noised to a random level (from almost clean to pure noise), and the model learns to predict what was added. The **timestep** represents how noisy the sample is.

We use `shifted_logit_normal` sampling, which samples timesteps with higher weight toward the middle of the noise range (not too clean, not too noisy) — this is where the most useful learning signal for appearance and identity lives. It also adjusts this distribution based on video length, since longer videos need more signal at high noise levels to learn global structure.

---

## Data Pipeline

Training never uses raw video directly. Everything is pre-encoded into compact representations ("latents") before training begins. This is done once and then the latents are reused for every training step, which is why preprocessing takes hours but training is relatively fast per step.

```
Raw video  → Video VAE encoder   → latents/         (video latents, ~99% smaller than raw video)
Text prompt → Gemma 12B LLM      → conditions/      (text embeddings — the model's understanding of your caption)
Raw audio  → Audio VAE encoder   → audio_latents/   (audio latents)
```

The preprocessed directory must have all three folders with matching filenames before training can start.

**Why pre-encode?** The VAE (encoder/decoder) and text encoder together use ~20GB of VRAM. By running them once offline, training only needs to load the transformer + LoRA — much more VRAM-efficient.

**Video shape constraints** come from the VAE's compression ratios:
- Spatial: 32× reduction (a 960×544 video becomes a 30×17 latent grid)
- Temporal: 8× reduction, plus the formula requires `frames % 8 == 1` (so 49, 97, etc. — not 48 or 96)

### Caption Strategy

LTX-2.3 uses **Gemma 3 12B** as its text encoder — a large language model that processes your caption into a rich semantic representation. The model was trained with very long, detailed captions (100–250 words describing both visual and audio content), so short prompts produce noticeably weaker conditioning.

Our captions combine Skye's transcript with the scene description:
```
SKYE says "Where are you going, silly goose?". A fixed medium close-up shot frames Skye, a golden-brown pup...
```
This tells the model simultaneously what Skye looks like, what she's doing, and what she's saying — all three dimensions we want the LoRA to learn.

---

## VRAM Requirements

| Setup | Peak VRAM | Who it's for |
|-------|-----------|--------------|
| Standard BF16, no quantization | ~80 GB | A100/H100 80GB |
| INT8 quantization (`int8-quanto`) | ~40–50 GB | A100 40GB |
| 8-bit text encoder only (`load_text_encoder_in_8bit`) | ~55–60 GB | Our current setup |

Our server has an 80GB GPU. The dry run peaked at **49.5 GB** with the 8-bit text encoder loaded during validation. During training proper (text encoder unloaded), it will be slightly less. We're well within limits.

---

## What Happens at Validation

Every N steps (set by `interval`), training pauses and the model generates a video from each validation prompt. This is your feedback loop:

- **Step 100–300:** Very early — characters look generic, but color palette and rough motion style may start to appear
- **Step 500–1000:** Character should be recognizable — pink/lavender coloring, correct body shape
- **Step 2000–2500:** Fine details — facial expressions, flight sequences, voice character

The validation videos are logged to W&B so you can watch the progression over time without having to download anything.

---

## Spatio-Temporal Guidance (STG)

STG is a technique used only at **inference time** (validation and production), not during training. It improves temporal coherence — preventing the generated video from flickering or losing consistency frame to frame.

How it works: the model runs two forward passes. In the second pass, it intentionally skips the self-attention in one specific block (block 29 out of 48). The difference between the two outputs is used as an extra "coherence signal" that gets blended back in. Think of it as the model comparing what it produces normally vs. what it produces when it deliberately ignores short-range temporal patterns — and using that contrast to reinforce consistency.

The recommended settings (from Lightricks testing):
```yaml
stg_scale: 1.0      # how strongly to apply the coherence boost
stg_blocks: [29]    # which block to perturb (mid-to-late = best empirically)
stg_mode: "stg_av"  # apply to both audio and video branches
```

You never need to tune these for character LoRA training.

---

## Why Not Train the VAE or Text Encoder?

**Video VAE:** Handles the compression and reconstruction of pixels. LoRA here would affect how sharp or artifact-free the video looks, not what the video *depicts*. Training this for a character is the wrong lever.

**Gemma text encoder:** Already rich at 12B parameters. Fine-tuning it risks "breaking" its general language understanding (it might forget what "helicopter" means in a broader context). The right approach is to train the transformer's cross-attention projections (`attn2.to_k`, `attn2.to_v`) — these are the interface between the text understanding (which stays frozen) and the video generation (which adapts). This is exactly what we're doing.

**BigVGAN vocoder:** Converts the model's internal audio representation to actual waveforms. LoRA here would affect audio quality/timbre at a low level. Character voice is better learned through the audio attention modules in the transformer.

---

## Quick Reference: Our Experiment Configs

| Config | Rank | Steps | Key Difference | When to Use |
|--------|------|-------|----------------|-------------|
| `skye_dryrun.yaml` | 16 | 100 | 10 clips only | ✅ Pipeline validation (done) |
| `skye_exp1_baseline.yaml` | 16 | 1000 | Fast check on full data | First real run — go/no-go |
| `skye_exp2_standard.yaml` | 32 | 2500 | Primary experiment | Main character LoRA |
| `skye_exp3_highcap.yaml` | 32 | 3000 | + FFN layers | Only if exp2 underfits |
| `skye_exp4_i2v.yaml` | 32 | 2500 | 80% image conditioning | Production I2V workflow |

---

## Appendix: Full Module Name Map (for reference)

Each of the 48 transformer blocks contains these LoRA-targeted layers (N = 0 to 47):

```
transformer_blocks.N.attn1.to_q              ← video self-attn: query
transformer_blocks.N.attn1.to_k              ← video self-attn: key
transformer_blocks.N.attn1.to_v              ← video self-attn: value
transformer_blocks.N.attn1.to_out.0          ← video self-attn: output projection
transformer_blocks.N.attn2.to_q/k/v/to_out.0  ← video cross-attn (to text)
transformer_blocks.N.audio_attn1.*            ← audio self-attn
transformer_blocks.N.audio_attn2.*            ← audio cross-attn (to text)
transformer_blocks.N.audio_to_video_attn.*    ← audio→video cross-modal
transformer_blocks.N.video_to_audio_attn.*    ← video→audio cross-modal
transformer_blocks.N.ff.net.0.proj            ← video FFN (optional LoRA)
transformer_blocks.N.ff.net.2                 ← video FFN (optional LoRA)
transformer_blocks.N.audio_ff.net.0.proj      ← audio FFN (optional LoRA)
transformer_blocks.N.audio_ff.net.2           ← audio FFN (optional LoRA)
```

Writing `target_modules: ["to_q", "to_k", "to_v", "to_out.0"]` in the config matches all of the non-optional rows above automatically — the framework does suffix matching across the entire model tree.

---

*All findings derived from the local codebase. Parameters are the official Lightricks defaults and should be treated as the authoritative starting point.*