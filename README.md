# Evaluating Test-Time Training for Vision Transformer-Based Robotic Grasp Detection

PyTorch implementation for the Forschungspraktikum report "Evaluating Test-Time Training for Vision Transformer-Based Robotic Grasp Detection" at the Chair of Robotics, Artificial Intelligence and Real-Time Systems, TUM.

This work evaluates replacing standard softmax attention with Test-Time Training (TTT) layers in a ViT-based robotic grasp detection model. A ViT-Small encoder is paired with a GR-ConvNet-style convolutional decoder and evaluated on the Cornell Grasp Dataset using depth-only input. The TTT variant substitutes all attention blocks with ViT³-style TTT layers, reducing attention complexity from O(N²) to O(N). TTT imposes overhead at small resolutions but becomes ~1.9× faster than the baseline at 1248×1248, with comparable grasp detection accuracy across both models.

This code is built on top of [TF-Grasp](https://github.com/WangShaoSUN/grasp-transformer) by Wang et al. The data loading pipeline, training loop, loss functions, and evaluation code are retained from that repository. See acknowledgements below.

---

## Requirements

Python 3.x. Install dependencies with:
```bash
pip install -r requirements.txt
```

Developed and tested on Ubuntu with CUDA 12.2 and an NVIDIA GeForce RTX 2080 Ti.

---

## Dataset

Download and extract the [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php).

---

## Training

Training uses 5-fold image-wise cross-validation on the Cornell dataset, depth-only input.

**TTT model (ViT with TTT attention blocks):**
```bash
python main_k_fold.py \
  --dataset cornell \
  --dataset-path ~/ethan/grasp-transformer/data/cornell \
  --use-depth 1 \
  --use-rgb 0 \
  --image-size 784 \
  --model vittttgrasp \
  --epochs 50 \
  --batch-size 2
```

**Baseline ViT model (standard softmax attention):**
```bash
python main_k_fold.py \
  --dataset cornell \
  --dataset-path ~/ethan/grasp-transformer/data/cornell \
  --use-depth 1 \
  --use-rgb 0 \
  --image-size 512 \
  --model vitgrasp \
  --epochs 100 \
  --batch-size 32 #Depends on image size and vram of the graphics card
```

`--image-size` must be a multiple of 16. Resolutions above 480 require upscaling the Cornell images and are suitable for efficiency benchmarking only; accuracy results should use under 480.

Trained models are saved in `output/models/` by default.

Included in the models folder are additional models which are not discussed in the paper but can be moved into the outer folder to be tested as well for comparison.

---

## Results

### Inference Time

| Resolution | Baseline ViT (ms) | ViT-TTT (ms) | TTT / Baseline |
|------------|-------------------|--------------|----------------|
| 224×224    | 5.5               | 16.1         | 2.93×          |
| 480×480    | 14.8              | 26.0         | 1.76×          |
| 784×784    | 47.8              | 40.0         | 0.84×          |
| 1024×1024  | 98.5              | 66.7         | 0.68×          |
| 1248×1248  | 170.0             | 88.6         | 0.52×          |

Crossover occurs between 480 and 784 pixels, consistent with the O(N) vs O(N²) complexity difference.

### Grasp Detection Accuracy

Image-wise IoU accuracy on Cornell Grasp Dataset, depth-only, 5-fold cross-validation. Best accuracy within 100 training epochs per fold.

| Model       | 224×224 | 480×480 |
|-------------|---------|---------|
| ViT baseline | 85.7%  | 93.8%   |
| ViT-TTT      | 95.1%  | 92.6%   |

---

## Acknowledgements

This codebase is built on [TF-Grasp](https://github.com/WangShaoSUN/grasp-transformer) (Wang et al., 2022). The data loading, training loop, loss functions, and evaluation code are taken directly from that repository.
```
@ARTICLE{9810182,
  author={Wang, Shaochen and Zhou, Zhangli and Kan, Zhen},
  journal={IEEE Robotics and Automation Letters},
  title={When Transformer Meets Robotic Grasping: Exploits Context for Efficient Grasp Detection},
  year={2022},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2022.3187261}
}
```

The TTT layer implementation follows the ViT³ design from Han et al. (2025), "ViT³: Unlocking Test-Time Training in Vision," arXiv:2512.01643.
