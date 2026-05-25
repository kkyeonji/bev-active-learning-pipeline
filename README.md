# BEVFormer Active Learning Pipeline

An end-to-end active learning pipeline built on top of [BEVFormer](https://github.com/fundamentalvision/BEVFormer) for camera-based 3D object detection in autonomous driving.

The core idea: instead of training on a fixed dataset, this pipeline **identifies which scenes the model struggles with**, clusters them by failure mode, and selects the most informative samples for the next training round — closing the loop between evaluation and data curation.

---

## What This Project Does

```
Initial Training          Evaluation           Scene Selection
─────────────────    ─────────────────    ─────────────────────────
BEVFormer-tiny    →  Per-scene metrics  →  Cluster hard scenes       →
nuScenes mini        (mAP, NDS, etc.)      by failure mode               │
                                                                          │
        ┌─────────────────────────────────────────────────────────────────┘
        ↓
New Data Config         Resume Training       Automated Orchestration
─────────────────    ─────────────────    ─────────────────────────
Subset + hard scenes →  Fine-tune from   →  Apache Airflow DAG
added to training       latest checkpoint    runs all steps end-to-end
```

---

## Pipeline Stages

### 1. Baseline Training
Train BEVFormer-tiny on nuScenes mini to establish a performance baseline.

### 2. Per-Scene Evaluation
Run evaluation per scene (not just dataset-level) to compute granular metrics: per-class AP, distance errors, velocity estimation errors. Scenes are scored by how far each metric falls below the population median.

### 3. Hard Scene Analysis & Clustering
Difficult scenes are analyzed and grouped by **failure mode** using feature-based clustering:
- **Scene-level features**: weather tags, time-of-day, object density, speed distribution
- **Error-level features**: missed detections vs. false positives, distance from ego, object category
- Clustering via k-means / DBSCAN to identify distinct hard-scene categories (e.g., "crowded intersection at night", "fast-moving distant objects")

### 4. Active Sample Selection
Given the cluster assignments, select scenes that maximize coverage of underrepresented failure modes — avoiding redundant sampling of already-seen scenarios. Selection strategies implemented:
- **Uncertainty-based**: scenes where prediction confidence is low
- **Diversity-based**: coverage sampling across clusters
- **Hybrid**: weighted combination of both

### 5. Data Config Generation & Resume Training
Automatically generate a new mmdet3d-compatible data config that merges the original training split with the selected hard scenes, then resume training from the latest checkpoint.

### 6. Orchestration with Apache Airflow
All five stages above are wired into an Airflow DAG. Each stage is a task; the pipeline can be triggered manually or on a schedule. Logs, metrics, and selected scene IDs are tracked at each step.

---

## Stack

| Component | Choice |
|---|---|
| Detection model | BEVFormer-tiny (R50 backbone) |
| Dataset | nuScenes mini |
| Training framework | mmdet3d / mmcv |
| Clustering | scikit-learn (k-means, DBSCAN) |
| Orchestration | Apache Airflow (local, Docker Compose) |
| Experiment tracking | *(planned: MLflow or W&B)* |

---

## Results

> Results will be updated as pipeline iterations complete.

| Round | Training Data | NDS | mAP | Notes |
|---|---|---|---|---|
| 0 (baseline) | nuScenes mini train split | — | — | In progress |
| 1 | Round 0 + selected hard scenes | — | — | Pending |

---

## Getting Started

### Prerequisites
- CUDA 11.x, Python 3.8+
- Docker & Docker Compose (for Airflow)

### Installation

```bash
git clone https://github.com/kkyeonji/bev-active-learning-pipeline.git
cd bev-active-learning-pipeline
pip install -r requirements.txt
```

See [docs/install.md](docs/install.md) for the full mmdet3d/mmcv environment setup.

### Dataset

Download the nuScenes mini split and run the data preparation script:

```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-mini
```

See [docs/prepare_dataset.md](docs/prepare_dataset.md) for details.

### Baseline Training

```bash
python tools/train.py projects/configs/bevformer/bevformer_tiny.py
```

### Per-Scene Evaluation

```bash
python tools/test.py projects/configs/bevformer/bevformer_tiny.py \
    <checkpoint.pth> \
    --eval bbox \
    --eval-options jsonfile_prefix=./results/round0
```

*(Active learning analysis scripts live in `active_learning/` — in progress)*

### Run the Full Pipeline (Airflow)

```bash
cd airflow
docker-compose up -d
# Open http://localhost:8080, trigger the `bev_active_learning` DAG
```

---

## Repository Structure

```
.
├── projects/configs/bevformer/   # mmdet3d model configs
├── tools/                        # train, test, data prep scripts
├── active_learning/              # scene scoring, clustering, selection (WIP)
│   ├── score_scenes.py
│   ├── cluster_scenes.py
│   └── select_scenes.py
├── airflow/                      # DAG definitions and Docker Compose (WIP)
│   ├── dags/bev_active_learning.py
│   └── docker-compose.yml
└── docs/
```

---

## Motivation

Large-scale annotation is expensive. Active learning helps answer: **which unlabeled (or under-trained) scenes are worth labeling next?** In practice, autonomous driving datasets contain a long tail of rare but safety-critical scenarios — heavy rain, dense pedestrians, unusual object orientations. A model trained without targeting these scenes will underperform on exactly the situations that matter most.

This project demonstrates a practical, automated loop to surface and prioritize those scenarios using only the model's own evaluation signal — no human-in-the-loop required for the selection step.

---

## Acknowledgement

Built on top of [BEVFormer](https://github.com/fundamentalvision/BEVFormer) (Li et al., ECCV 2022). The original model architecture and training code are from the BEVFormer authors.

```
@article{li2022bevformer,
  title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2203.17270},
  year={2022}
}
```
