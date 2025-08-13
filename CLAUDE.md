# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is OC-SORT (Observation-Centric SORT), a multi-object tracking system built on YOLOX detection and advanced tracking algorithms. The codebase includes:

- **OC-SORT Tracker**: Main tracking algorithm with observation-centric re-update mechanism
- **DeepSORT Tracker**: Alternative tracker with appearance features
- **YOLOX Detection**: Object detection backbone
- **Multiple Dataset Support**: MOT17, MOT20, DanceTrack, KITTI, CrowdHuman, etc.

## Key Commands

### Installation and Setup
```bash
# Install dependencies
pip3 install -r requirements.txt
python3 setup.py develop

# Additional dependencies
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox pandas xmltodict
```

### Training
```bash
# Train on MOT17 (main model)
python3 tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth

# Train on DanceTrack
python3 tools/train.py -f exps/example/mot/yolox_dancetrack_test.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

### Evaluation and Inference
```bash
# Demo tracking on video
python3 tools/demo_track.py --demo_type video -f exps/example/mot/yolox_dancetrack_test.py -c pretrained/ocsort_dance_model.pth.tar --path videos/dance_demo.mp4 --fp16 --fuse --save_result

# Evaluate on MOT17
python3 tools/run_ocsort.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse

# Evaluate on DanceTrack
python tools/run_ocsort_dance.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/bytetrack_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse
```

### Data Conversion
```bash
# Convert datasets to COCO format
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_dance_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
```

### Post-processing
```bash
# Linear interpolation for offline tracking
python3 tools/interpolation.py $result_path $save_path

# Gaussian Process interpolation
python3 tools/gp_interpolation.py $raw_results_path $linear_interp_path $save_path
```

## Architecture Overview

### Core Tracking Components
- **trackers/ocsort_tracker/**: OC-SORT implementation with observation-centric re-update
- **trackers/deepsort_tracker/**: DeepSORT with appearance features
- **trackers/byte_tracker/**: ByteTrack baseline implementation

### Detection Framework
- **yolox/**: YOLOX detection framework (models, data loading, training)
- **exps/**: Experiment configurations for different datasets and model sizes

### Evaluation and Metrics
- **trackeval/**: Tracking evaluation toolkit
- **motmetrics/**: MOT challenge metrics implementation

### Key Files
- `tools/demo_track.py`: Main demo script for video tracking
- `tools/run_ocsort.py`: OC-SORT evaluation on MOT datasets
- `tools/train.py`: Training script for detection models
- `trackers/ocsort_tracker/ocsort.py`: Core OC-SORT algorithm
- `trackers/ocsort_tracker/association.py`: Data association logic

## Dataset Structure
Expected dataset layout under `datasets/`:
```
datasets/
├── mot/               # MOT17 data
├── MOT20/             # MOT20 data  
├── dancetrack/        # DanceTrack data
├── crowdhuman/        # CrowdHuman data
└── Cityscapes/        # Cityperson data
```

## Model Weights
Pre-trained models should be placed in `pretrained/`:
- YOLOX backbone weights (from YOLOX model zoo)
- Fine-tuned tracking models for specific datasets
- Detection models trained on mixed datasets

## Deployment Options
- **ONNX**: Export via `deploy/scripts/export_onnx.py`
- **TensorRT**: Python and C++ support available
- **ncnn**: Mobile deployment support
- **C++**: Full C++ implementation in `deploy/OCSort/cpp/`