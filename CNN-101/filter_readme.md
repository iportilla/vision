# CNN Filters — How Are the Values Assigned?

> A filter (kernel) is the heart of a convolutional layer.  
> This document explains **where the numbers come from** — both the hand-crafted approach used in `CNN-101.ipynb` and the learned approach used in real neural networks.

---

## Two Ways to Set Filter Values

```mermaid
flowchart TD
    Q["How are filter values assigned?"]
    Q --> H["Hand-crafted\n(Classical signal processing)"]
    Q --> L["Learned\n(Backpropagation)"]

    H --> H1["Human designer picks values\nbased on maths / intuition"]
    H --> H2["Example: Sobel, Laplacian,\nGaussian blur"]
    H --> H3["Fixed — they never change\nduring training"]

    L --> L1["Values start as random\nsmall numbers"]
    L --> L2["Network adjusts them\nautomatically during training"]
    L --> L3["End result: filters that\nmaximise task accuracy"]
```

---

## Approach 1 — Hand-Crafted Filters

The filters in `CNN-101.ipynb` are manually designed using well-known formulas from **image processing**.

### The Sobel Y filter (used in the notebook)

```
[[-1, -2, -1],
 [ 0,  0,  0],
 [ 1,  2,  1]]
```

The values are chosen so the filter computes an **approximation of the vertical image gradient** — the difference in brightness between the row above and the row below each pixel.

```mermaid
flowchart LR
    subgraph Top["Top row — weight −1, −2, −1"]
        T["Bright pixels ABOVE\nthe current position\ncontribute negatively"]
    end
    subgraph Mid["Middle row — weight 0, 0, 0"]
        M["Current row ignored\n(only the difference matters)"]
    end
    subgraph Bot["Bottom row — weight +1, +2, +1"]
        B["Bright pixels BELOW\ncontribute positively"]
    end

    Top --> Out["Large positive output\n→ dark-above / bright-below edge"]
    Mid --> Out
    Bot --> Out
    Out --> C["Clamp to 0–255\n→ bright pixel in feature map"]
```

The centre column is weighted **double** (−2 / +2) compared to the corners (−1 / +1) so that horizontally adjacent pixels matter more than diagonal ones.

### Other hand-crafted examples

```mermaid
flowchart LR
    SX["Sobel X\n[[-1,0,1],[-2,0,2],[-1,0,1]]\nVertical edges\n(left–right gradient)"]
    SY["Sobel Y\n[[-1,-2,-1],[0,0,0],[1,2,1]]\nHorizontal edges\n(top–bottom gradient)"]
    LAP["Laplacian\n[[0,1,0],[1,-4,1],[0,1,0]]\nAll edges\n(second derivative)"]
    BLUR["Gaussian blur\n[[1,2,1],[2,4,2],[1,2,1]] ÷ 16\nSmooth / denoise"]

    SX --> FM["Feature maps\n(one per filter)"]
    SY --> FM
    LAP --> FM
    BLUR --> FM
```

---

## Approach 2 — Learned Filters (Backpropagation)

In a trained CNN (e.g. VGG-16, ResNet), **no human picks the filter values**. They are initialised randomly and gradually improved by backpropagation.

### Training loop

```mermaid
flowchart LR
    INIT["Random init\ne.g. values ~ N(0, 0.01)"]
    --> FW["Forward pass\nConvolve image with current K"]
    --> PRED["Prediction\ne.g. 'cat' with 23% confidence"]
    --> LOSS["Compute loss\nL = cross_entropy(pred, true_label)"]
    --> GRAD["Backpropagation\nCompute ∂L/∂K for every\nfilter weight"]
    --> UPD["Update weights\nK ← K − η · ∂L/∂K"]
    --> UPD2{"Converged?"}
    UPD2 -- No --> FW
    UPD2 -- Yes --> DONE["Learned filter\nDetects a specific feature\nuseful for the task"]
```

$\eta$ is the **learning rate** — a small number (e.g. 0.001) that controls how big each update step is.

### The update rule in full

$$K_{ij} \;\leftarrow\; K_{ij} \;-\; \eta \;\frac{\partial \mathcal{L}}{\partial K_{ij}}$$

| Symbol | Meaning |
|---|---|
| $K_{ij}$ | One weight in the filter at position $(i, j)$ |
| $\mathcal{L}$ | Loss — how wrong the prediction was |
| $\frac{\partial \mathcal{L}}{\partial K_{ij}}$ | Gradient — which direction to nudge this weight |
| $\eta$ | Learning rate — how big the nudge is |

---

## What Do Learned Filters End Up Detecting?

```mermaid
flowchart TD
    IN["Input image"]

    subgraph L1["Layer 1 filters"]
        L1A["Edges"]
        L1B["Colours"]
        L1C["Orientations\n(0°, 45°, 90°)"]
    end

    subgraph L2["Layer 2 filters"]
        L2A["Corners"]
        L2B["Textures\n(fur, bricks)"]
        L2C["Blobs / spots"]
    end

    subgraph L3["Layer 3+ filters"]
        L3A["Object parts\n(eyes, wheels, noses)"]
        L3B["Repeated patterns"]
    end

    subgraph DEEP["Deep layers"]
        DA["Whole objects\n(faces, cars)"]
        DB["Semantic concepts"]
    end

    IN --> L1
    L1 --> L2
    L2 --> L3
    L3 --> DEEP
```

> **Key insight:** Layer 1 filters in a trained CNN almost always look like **Sobel / Gabor filters** — the same patterns that signal-processing engineers derived by hand decades earlier. Backprop rediscovers them automatically because edges really are the most informative low-level feature in natural images.

---

## Hand-Crafted vs Learned — Summary

```mermaid
flowchart LR
    subgraph HC["Hand-Crafted"]
        HC1["✅ Interpretable — you know what they detect"]
        HC2["✅ No training data needed"]
        HC3["✅ Fast — no training loop"]
        HC4["❌ Limited — can only detect\nwhat a human can describe"]
        HC5["❌ Does not adapt to the task"]
    end

    subgraph LR2["Learned"]
        LR1["✅ Adapts to the task automatically"]
        LR2b["✅ Discovers features humans\nnever thought of"]
        LR3["✅ Scales to deep networks\nwith millions of filters"]
        LR4["❌ Needs lots of labelled data"]
        LR5["❌ Harder to interpret\n(black box)"]
    end
```

---

## How This Notebook Fits In

`CNN-101.ipynb` uses the **hand-crafted** approach to keep the focus on *understanding the mechanism* of convolution — the maths is identical whether the values were hand-designed or learned.

In a real framework (PyTorch / TensorFlow) you would do:

```python
import torch.nn as nn

# Define a conv layer — PyTorch randomly initialises the filter
# and learns the values during model.fit() / training loop
conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

# After training, inspect what a filter learned:
print(conv.weight[0])   # first filter's 3×3 values
```

The rest — the sliding window, the multiply-and-sum, the feature map — is exactly what the notebook implements by hand.
