## Checkpoint of our experiments

We provide our checkpoints reproduced by this repo as follows:

### Image tasks

| Model         | Dataset| Epoch | r (num_prune)  | Top-1 Acc. (%) | FLOPs(G) | Download |  Log |
|---------------| ------------ |-------|----------------|----------------|----------|-------- |-------- |
| Vim-Ti        | ImageNet-1K | 30    | 5              | 75.3           | 1.28     | [:hugs:HF](https://huggingface.co/Soptq/R-Meeto/tree/main/Vim-Ti_30epoch_75.3)| [🔗](https://huggingface.co/Soptq/R-Meeto/tree/main/Vim-Ti_30epoch_75.3/logs)|
| Vim-S         | ImageNet-1K | 15    | 11             | 79.9           | 3.58     | [:hugs:HF](https://huggingface.co/Soptq/R-Meeto/tree/main/Vim-S_15epoch_80.0)| [🔗](https://huggingface.co/Soptq/R-Meeto/tree/main/Vim-S_15epoch_80.0/logs)|
| Vim-B         | ImageNet-1K | 15    | 11             | 81.3           | 13.21    | [:hugs:HF](https://huggingface.co/Soptq/R-Meeto/tree/main/Vim-B_15epochs_81.3)| [🔗](https://huggingface.co/Soptq/R-Meeto/tree/main/Vim-B_15epochs_81.3/logs)|
| VideoMamba-Ti | ImageNet-1K | 30    | 5              | 75.9           | 1.28     | [:hugs:HF](https://huggingface.co/Soptq/R-Meeto/tree/main/VideoMamba-Ti_30epoch_75.9)| [🔗](https://huggingface.co/Soptq/R-Meeto/tree/main/VideoMamba-Ti_30epoch_75.9/logs)|
| VideoMamba-S  | ImageNet-1K | 15    | 11             | 80.1           | 3.58     | [:hugs:HF](https://huggingface.co/Soptq/R-Meeto/tree/main/VideoMamba-S_15epochs_80.1)| [🔗](https://huggingface.co/Soptq/R-Meeto/tree/main/VideoMamba-S_15epochs_80.1/logs)|
| VideoMamba-B  | ImageNet-1K | 15    | 11             | 81.9           | 13.21    | [:hugs:HF](https://huggingface.co/Soptq/R-Meeto/tree/main/VideoMamba-B_15epochs_81.9)| [🔗](https://huggingface.co/Soptq/R-Meeto/tree/main/VideoMamba-B_15epochs_81.9/logs)|

### Video Task

| Model         | Dataset | Epoch | r (num_prune) | Top-1 Acc. (%) | FLOPs(G) | Download | Log                                                                                           |
|---------------|---------|-------|---------------|----------------|----------|-------- |-----------------------------------------------------------------------------------------------|
| VideoMamba-Ti | K400    | 30    | 88            | 76.5           | 9.44     | [:hugs:HF](https://huggingface.co/Soptq/R-Meeto/tree/main/K400_VideoMamba-Ti_30epochs_76.5)| [🔗](https://huggingface.co/Soptq/R-Meeto/blob/main/K400_VideoMamba-Ti_30epochs_76.5/log.txt) |
| VideoMamba-S  | K400    | 15    | 88            | 78.5           | 30.99    | [:hugs:HF](https://huggingface.co/Soptq/R-Meeto/tree/main/K400_VideoMamba-S_15epochs_78.5)| [🔗](https://huggingface.co/Soptq/R-Meeto/tree/main/K400_VideoMamba-S_15epochs_78.5/log.txt)  |
| VideoMamba-B  | K400    | 15    | 88            | 78.9           | 67.11    | [:hugs:HF](https://huggingface.co/Soptq/R-Meeto/tree/main/K400_VideoMamba-M_15epochs_78.9)| [🔗](https://huggingface.co/Soptq/R-Meeto/tree/main/K400_VideoMamba-M_15epochs_78.9/log.txt)  |
