# Patch Policy reproduction

This repository contains a self-contained reproduction harness for the central
architectural claim in *Patch Policy: Efficient Embodied Control via Dense
Visual Representations* (arXiv:2607.18236).

The first benchmark is a controlled visuomotor task: a frozen DINOv2 ViT-S/14
encodes short image sequences containing an effector, a goal, and distractors.
A causal transformer predicts the goal-directed control vector. Experiment
branches vary only the committed `config.json`, while `bash run.sh` remains the
fixed run contract. Eight independent seeds run in parallel on the eight GPUs
of one Kubernetes node.

The runner prints per-seed metrics and a final `ORX_RESULT` JSON record. If the
public DINOv2 checkpoint cannot be downloaded, it uses a deterministic frozen
patch projection and clearly records that fallback; such a run validates the
architecture but not the pretrained-representation claim.
