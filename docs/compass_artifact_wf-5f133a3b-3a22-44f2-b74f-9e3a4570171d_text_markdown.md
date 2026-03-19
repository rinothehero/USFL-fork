# How FL and SFL papers handle ViT input resolution

**The dominant practice in federated learning research is straightforward: resize small images to 224×224.** Across virtually every FL and SFL paper published between 2022 and 2025 that uses standard ViT or DeiT models, CIFAR-10/100 images (natively 32×32) are upsampled to 224×224 via bilinear interpolation to match ImageNet-pretrained model expectations. This approach inflates per-image computation by roughly **49×**, yet almost no papers systematically analyze this overhead or explore alternatives within FL settings. A small number of papers use native-resolution inputs with modified architectures, and compact ViT variants like CCT remain strikingly under-explored in federated contexts despite being purpose-built for small images.

---

## Resizing to 224×224 is the near-universal default

The overwhelming majority of FL/SFL papers that pair ViT/DeiT models with small-image datasets simply resize inputs to 224×224 and use standard patch size 16×16 (producing 196 patches). This choice is tightly coupled to a second near-universal decision: **using ImageNet-pretrained weights**. Because pretrained positional embeddings encode spatial structure at 224×224 resolution, deviating from this input size would require interpolating or discarding those embeddings, undermining the transfer learning benefit.

The most explicit statement comes from **FES-PIT** (arXiv 2024), a federated split learning paper that evaluates the widest range of ViT/DeiT variants in any FL study. It states directly: *"Since the original resolutions of CIFAR-100 are too small for ViTs, we resize the input images to 224×224 (training and testing) while not modifying the ViT architectures."* FES-PIT tests ViT-Tiny, ViT-Small, DeiT-Tiny, DeiT-Small, DeiT-Base, and their distilled variants (DDeiT-T/S/B) on CIFAR-10, CIFAR-100, and Tiny-ImageNet (64×64, also upsampled to 224×224), achieving **96.87% on CIFAR-10** under Dirichlet-0.3 non-IID partitioning — far exceeding CNN baselines.

The foundational **ViT-FL paper** (Qu et al., CVPR 2022) established this pattern by using `vit_tiny_patch16_224`, `vit_small_patch16_224`, and `vit_base_patch16_224` from the timm library on CIFAR-10, with all models pretrained on ImageNet-1K. The paper demonstrated that simply swapping CNNs for ViTs in FedAvg improved accuracy by up to **77.7%** on extreme non-IID splits, showing ViTs are naturally more robust to data heterogeneity. Other papers following this exact approach include **FedTP** (IEEE TNNLS 2023) with ViT on CIFAR-10/100, **EFTViT** (ICCV 2025) with ViT-Base pretrained on ImageNet-21K on CIFAR-10/100, and **HePCo** (arXiv 2023) with ViT-B/16 on CIFAR-100.

In the split learning literature specifically, **CutMixSL** (FL-IJCAI 2022) and its differential privacy extension **DP-CutMixSL** (NeurIPS 2022 Workshop) both use ViT on CIFAR-10 at 224×224, splitting after the patch embedding layer. The recent **ADC paper** (arXiv 2025) uses DeiT-Tiny and DeiT-Small on CIFAR-100 and Food101, again at 224×224, with variable split points across transformer encoder blocks.

## Concrete paper-by-paper experimental details

The table below consolidates every FL/SFL paper found that uses ViT/DeiT variants with specific dataset, resolution, and architecture details:

| Paper | Venue | FL Type | ViT/DeiT Variant | Dataset | Input Size | CIFAR Handling |
|-------|-------|---------|-------------------|---------|------------|----------------|
| ViT-FL (Qu et al.) | CVPR 2022 | FL | ViT-T, ViT-S, ViT-B | CIFAR-10, Retina, CelebA | 224×224 | Resized from 32×32 |
| FedTP (Li et al.) | IEEE TNNLS 2023 | FL | ViT (standard) | CIFAR-10, CIFAR-100 | 224×224 | Resized from 32×32 |
| EFTViT (Wu et al.) | ICCV 2025 | Hierarchical FL | ViT-B (ImageNet-21K) | CIFAR-10, CIFAR-100, UC Merced | 224×224 | Resized from 32×32 |
| FES-PIT | arXiv 2024 | FSL | ViT-T/S, DeiT-T/S/B, DDeiT-T/S/B | CIFAR-10, CIFAR-100, Tiny-ImageNet | 224×224 | Explicitly resized |
| CutMixSL (Baek et al.) | FL-IJCAI 2022 | SL | ViT (DeiT-like) | CIFAR-10 | 224×224 | Resized from 32×32 |
| DP-CutMixSL (Oh et al.) | NeurIPS 2022 WS | SL | ViT | CIFAR-10 | 224×224 | Resized from 32×32 |
| ADC (Alvetreti et al.) | arXiv 2025 | SL | DeiT-Tiny, DeiT-Small | CIFAR-100, Food101 | 224×224 | Resized from 32×32 |
| HePCo | arXiv 2023 | FL | ViT-B/16 | CIFAR-100 | 224×224 | Resized from 32×32 |
| Zuo et al. | MLMI/ACM 2022 | FL | Modified ViTs | CIFAR-10, Fashion-MNIST | **Native (32×32)** | Not resized |
| FeSTA (Park et al.) | NeurIPS 2021 | SFL | Hybrid CNN+ViT | Medical CXR | 224×224 | N/A (medical) |
| FeSViBS (Almalik et al.) | MICCAI 2023 | SFL | Hybrid CNN+ViT-Base | Medical imaging | 224×224 | N/A (medical) |
| FedVKD | MDPI Electronics 2022 | FL+KD | CNN (client) + ViT (server) | CIFAR-10/100, ImageNet | 224×224 | Hybrid approach |

A few patterns stand out. Every paper using **pretrained** ViT/DeiT models resizes to 224×224 without exception. The only paper that operates at native resolution (Zuo et al., MLMI 2022) trains from scratch with modified architectures. Medical imaging papers sidestep the issue entirely since clinical images are naturally larger.

## The rare alternative: native resolution with modified architectures

Only one FL paper — **"An Empirical Analysis of Vision Transformer and CNN in Resource-Constrained Federated Learning"** (Zuo et al., MLMI/ACM 2022) — explicitly uses low-resolution images at their native size. This paper investigates whether ViTs retain advantages over CNNs when input resolution is kept small to satisfy edge-device resource constraints. Its key finding: **ViTs still achieve better global test accuracy than CNNs at low resolution** with comparable training cost, making them suitable for resource-constrained FL.

Outside FL, a rich body of work exists on adapting ViTs for 32×32 images. The standard modification is reducing **patch size from 16 to 4**, which produces 64 patches from a 32×32 image (8×8 grid) — enough for meaningful self-attention. The widely-cited `vision-transformers-cifar10` repository by kentaroy47 (referenced in 30+ CVPR/ICLR/NeurIPS papers) implements this exact approach. The paper "How to Train Vision Transformer on Small-scale Datasets?" (BMVC 2022) uses `image_size=32, patch_size=4` and combines it with self-supervised pretraining. More recently, "Powerful Design of Small Vision Transformer on CIFAR10" (arXiv 2025) achieves **93.58% on CIFAR-10** with a tiny ViT using patch_size=4, 12 attention heads, and 9 transformer blocks at native 32×32.

However, training from scratch with modified patch sizes yields substantially lower accuracy than using ImageNet-pretrained models at 224×224. A vanilla ViT trained from scratch on CIFAR-10 with patch_size=4 typically achieves only **78–80%**, compared to **96–98%** with pretrained weights and resizing. This accuracy gap explains why FL researchers overwhelmingly prefer the resize approach despite its computational cost.

## Compact ViT variants: a conspicuous gap in FL research

**Compact Convolutional Transformers (CCT)**, introduced by Hassani et al. (2021), were designed precisely for this resolution problem. CCT replaces the linear patch embedding with a convolutional tokenizer, makes positional embeddings optional, and uses sequence pooling instead of a class token. On CIFAR-10, CCT achieves **98% accuracy with fewer than 4M parameters** — vastly outperforming vanilla ViT (78.2% with 4.7M params) while working directly on 32×32 inputs. Similarly, **ViT-Lite** operates at native resolution with minimal parameters.

Despite these advantages, **no FL paper was found that directly uses CCT or ViT-Lite**. The MLMI 2022 paper by Zuo et al. cites CCT as related work but does not evaluate it. This represents a clear research gap. CCT's properties — small parameter count (**as few as 0.28M** vs. 86M for ViT-B), no need for large-scale pretraining, native 32×32 operation — align perfectly with FL's constraints around communication efficiency, client compute budgets, and training data limitations.

## Computational overhead: acknowledged but rarely analyzed

The computational cost of resizing 32×32 images to 224×224 is substantial. Pixel count increases by **49×** (from 3,072 to 150,528 values per image), and the quadratic self-attention cost scales with the number of patches: **196 patches at 224×224 versus just 64 at 32×32 with patch_size=4**, representing a roughly **9.4× increase** in attention computation alone.

**EFTViT** (ICCV 2025) provides the most detailed computational analysis in an FL context, though it addresses the cost through masking rather than questioning the resize decision. By randomly masking **75% of patches** after the 224×224 input is tokenized, EFTViT reduces computation by **5.2×** while maintaining near-full accuracy (90.02% vs. 90.40% on CIFAR-100). This implicitly acknowledges that processing all 196 patches from an upsampled 32×32 image is wasteful — most of those patches carry redundant interpolated information.

In split learning specifically, the resolution choice has direct implications for **communication cost**. ViT's patch token representations after the embedding layer are nearly as large as the input itself, unlike CNNs where pooling progressively reduces spatial dimensions. The CutMixSL and DP-CutMixSL papers address this by masking 20–50% of patch tokens before transmitting smashed data from client to server, reducing privacy leakage and bandwidth simultaneously.

No paper was found that **systematically compares** the three approaches (resize to 224×224, modify patch size for native resolution, use compact variants like CCT) within a unified FL experimental framework. No paper evaluates whether the accuracy gains from pretrained-model resizing justify the computational cost relative to compact alternatives, particularly in resource-constrained federated settings.

## Conclusion

The FL/SFL literature's treatment of ViT input resolution reveals a strong and largely unexamined convention. **Resizing small images to 224×224 to leverage ImageNet pretraining is the default** — adopted by every major FL paper using standard ViT/DeiT models on CIFAR-10/100. This approach delivers superior accuracy (often 96%+) but imposes significant computational overhead that particularly disadvantages resource-constrained clients in cross-device FL. The lone exception (Zuo et al., MLMI 2022) demonstrates that ViTs can outperform CNNs even at native low resolution, suggesting the resize-by-default approach deserves more scrutiny. The most actionable gap in the literature is the complete absence of compact transformers (CCT, ViT-Lite) from FL benchmarks — models that achieve near-state-of-the-art accuracy on small images at a fraction of the computational and communication cost. For researchers designing FL/SFL experiments with ViTs, the practical takeaway is clear: if using pretrained DeiT-Tiny/Small or ViT variants, resize to 224×224 as every existing paper does; but if training from scratch or operating under tight resource constraints, modifying patch size to 4 on native 32×32 inputs or adopting CCT-family models are viable and under-explored alternatives.