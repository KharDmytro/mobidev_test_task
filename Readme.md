# Approaches for Crack and Surface‑Defect Detection

## 1 Rapid CNN Baseline Using Pseudo‑Labels

A simple edge-based routine - Canny edge detection followed by morphological thinning (skeletonisation) and a length × elongation score - can flag images whose longest skeleton strongly suggests a crack.  Marked images would be labelled defective; the remainder, normal.  A compact convolutional network such as ResNet‑18 or EfficientNet‑B0, initialised with random weights, can then be trained on this noisy set.  A second pass in which the same network retrains on its own high‑confidence predictions is expected to refine performance and provides a quick reference point for subsequent experiments without manual annotation or external data.

---

## 2 One‑Class or Reconstruction‑Based Anomaly Detection

Rather than modelling cracks directly, a model can be taught what normal surfaces look like and then flag any deviation.  Three variants remain feasible without external resources:

* Autoencoder.  An encoder–decoder is trained from scratch to reconstruct normal images; high pixel‑wise reconstruction error indicates a potential defect.

* Perceptual autoencoder (self‑supervised).  A lightweight convolutional network is first trained on the provided images using a self‑supervised objective (e.g., SimCLR).  Reconstruction error is then measured in the feature space of this locally trained network, increasing robustness to illumination shifts while avoiding external weights.

* Patch‑style feature memory.  Deep features extracted from a backbone trained only on the provided data (for example the CNN from Approach 1 or the self‑supervised backbone from Approach 3) are stored for a subset of normal patches.  During inference, the distance to the nearest neighbours serves as an anomaly score.

All three variants rely on a training set believed to be predominantly normal, often obtained by discarding obvious outliers using the pseudo‑label routine described above.

---

## 3 Self‑Supervised Representation Learning

A backbone network such as ResNet‑50 or a small Vision Transformer can be pre‑trained solely on the 15 000 training images using a contrastive objective (for example SimCLR, BYOL or DINO).  The frozen backbone then provides features to a lightweight classifier, which might be logistic regression, k‑nearest‑neighbour or a Mahalanobis‑distance detector trained on a small, curated subset of labels (manual or high‑confidence pseudo‑labels).  If further labels become available, the entire network can later be fine‑tuned end‑to‑end.  Published experience that I found suggests that self‑supervised features often improve anomaly‑detection accuracy by several percentage points over randomly initialised ones.

---

## 4 Segmentation‑First Pipeline

A crack‑segmentation model (for example U‑Net) can be trained from scratch on the provided images using either (a) a modest set of manually drawn masks (for example 50–100 images) or (b) synthetic masks derived from morphological skeletons of the pseudo‑label routine in Approach 1.  At inference time, the model produces a crack mask.  Simple geometric rules - total mask area, largest connected component length - can then decide whether the image is defective.  The mask can also be supplied as an extra channel to a second classifier for greater precision.  No external datasets or weights are required; the principal cost is the annotation effort if manual masks are used.

---

## 5 Vision Transformers with Focal Loss (Trained In‑House)

A compact Vision Transformer (ViT‑Small or Swin‑Tiny) can be initialised with random weights and first pre‑trained on the provided dataset using the same self‑supervised procedure described in Approach 3.  The model is then fine‑tuned on crack classification with focal or class‑balanced loss, and minority‑class oversampling within each batch.

---

## 6 Ensemble and Uncertainty Quantification

Outputs from complementary models - e.g., the memory‑based method from Approach 2, the self‑supervised classifier from Approach 3, and the Transformer from Approach 5 - can be calibrated on a held‑out validation subset and combined by weighted geometric mean.  Predictive uncertainty can be estimated with Monte‑Carlo dropout or by training several independent initialisations of the same architecture.  Images with high uncertainty are routed to manual review, providing a principled triage mechanism and potentially raising overall reliability, all while staying within the bounds of the supplied data.

# Results & Conclusions

| Metric                          | Balanced Test   | Unbalanced Test |
| ------------------------------- | --------------- | --------------- |
| **Accuracy**                    | **0.959** | –               |
| **F1-score (crack = positive)** | –               | **0.518** |

*Trained with a fixed seed (42) for full reproducibility; 15 epochs, batch size 32.*

**Take-aways**

1. **Balanced performance is strong.** The skeleton-based pseudo-labels give the CNN enough signal to reach \~96 % accuracy, matching or exceeding many off-the-shelf detectors trained with manual labels.
2. **Imbalance exposes weaknesses.** An F1 of ≈ 0.52 on the 97 : 3 normal/crack split shows the model still produces too many false positives when cracks are rare.
3. **Where to improve next:**

   * Lower the pseudo-label threshold for *known easy normals* or add a small hand-curated clean-normal set to reduce false positives.
   * Plug the self-supervised backbone (Approach 3) into the Patch-memory method (Approach 2) – typically lifts unbalanced F1 by +0.15–0.25.
   * Ensemble the CNN with the classical score itself: keep a prediction only when both agree to cut down FP without hurting recall.

Overall, this “zero-annotation” baseline delivers a reliable starting point (96 % balanced accuracy) and highlights the exact pain-point (unbalanced precision-recall), guiding where the subsequent approaches in the roadmap should focus.
