## Baseline pretrained model 

#### For image task:
| Model    | Dataset| Top-1 Acc. (%)  | Download  |
| -------- | ------------ | ------------  | -------- |
| Vim-Ti | ImageNet-1K | 76.1 |[🔗 ](https://huggingface.co/hustvl/Vim-tiny-midclstok)|
| Vim-S | ImageNet-1K | 80.5 |[🔗 ]( https://huggingface.co/hustvl/Vim-small-midclstok)|
| Vim-B | ImageNet-1K | 81.9(EMA) |[🔗 ](https://huggingface.co/hustvl/Vim-base-midclstok)|
| VideoMamba-Ti| ImageNet-1K | 76.9 |[🔗 ](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_in1k_res224.pth)|
| VideoMamba-S| ImageNet-1K | 81.2 |[🔗 ](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_in1k_res224.pth)|
| VideoMamba-B| ImageNet-1K | 82.7 |[🔗 ](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_b16_in1k_res224.pth)|
| DeiT-Ti | ImageNet-1K | 72.2 | [🔗 ](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| DeiT-S | ImageNet-1K | 79.8 |[🔗 ]( https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)|
| DeiT-B | ImageNet-1K | 81.8 |[🔗 ](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)|


#### For video task:
| Model    | Dataset|#Frame| Top-1 Acc. (%)  | Download |
| -------- | ------------ | ------------  | -------- | -------- |
| VideoMamba-Ti| K400 |8x3x4   | 76.9 |[ 🔗](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_t16_in1k_res224.pth)|
| VideoMamba-S| K400 | 8x3x4   | 79.3 |[ 🔗](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_s16_in1k_res224.pth)|
| VideoMamba-M| K400 | 8x3x4   | 80.6 |[ 🔗](https://huggingface.co/OpenGVLab/VideoMamba/blob/main/videomamba_b16_in1k_res224.pth)|