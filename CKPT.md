## Checkpoint of our experiments

We provide our checkpoints reproduced by this repo as follows:

### Image tasks

| Model         | Dataset| Epoch | r (num_prune)  | Top-1 Acc. (%) | FLOPs(G) | Download |  Log |
|---------------| ------------ |-------|----------------|----------------|----------|-------- |-------- |
| Vim-Ti        | ImageNet-1K | 30    | 5              | 75.3           | 1.28     | [:hugs:HF](https://huggingface.co/Aristo2333/R-MeeTo/blob/main/Table2/Vim-Ti_30epoch_75.3.pth)| [🔗](https://drive.google.com/drive/folders/1Inp8m5D5BzuOdxmlvp8ivvjcZpWBDTlO?usp=drive_link)|
| Vim-S         | ImageNet-1K | 15    | 11             | 79.9           | 3.58     | [:hugs:HF](https://huggingface.co/Aristo2333/R-MeeTo/resolve/main/Table2/Vim-S_15epoch_80.0.pth?download=true)| [🔗](https://drive.google.com/drive/folders/19nup9W9NHPl7dSa0P0xC-poqOe8xsxqu?usp=drive_link)|
| Vim-B         | ImageNet-1K | 15    | 11             | 81.3           | 13.21    | [🔗](https://drive.google.com/file/d/1bF5XNzVYY0sTRuByz5G9GJJIbae0E9iz/view?usp=sharing)| [🔗](https://drive.google.com/drive/folders/1J3UZOvScq-ayiKe5PFFe6l1f4kfKO55y?usp=sharing)|
| VideoMamba-Ti | ImageNet-1K | 30    | 5              | 75.9           | 1.28     | [🔗](https://drive.google.com/file/d/1qWmqwnQ0WiaLObDY_suJN4pjtZFTkhVS/view?usp=sharing)| [🔗](https://drive.google.com/drive/folders/1N4mgVdumd0Lb0smLp644B6mHOSHNwce6?usp=sharing)|
| VideoMamba-S  | ImageNet-1K | 15    | 11             | 80.1           | 3.58     | [🔗](https://drive.google.com/file/d/1RV5Fv3LGqZqp9Lodpc4mdTIqlV6XBt58/view?usp=sharing)| [🔗](https://drive.google.com/drive/folders/1ffprbkeOK2NOf6nmXukrMVJ8wvyDv65r?usp=sharing)|
| VideoMamba-B  | ImageNet-1K | 15    | 11             | 81.9           | 13.21    | [🔗](https://drive.google.com/file/d/1FxTrga6pI7ZPvmJ5s87FijlyFe6-53E3/view?usp=sharing)| [🔗](https://drive.google.com/drive/folders/1BYoUM8K9-WNYuQHxT2avKLY2l2ZdLqXU?usp=sharing)|

### Video Task
| Model         | Dataset | Epoch | r (num_prune) | Top-1 Acc. (%) | FLOPs(G) | Download |  Log |
|---------------|---------|-------|---------------|----------------|----------|-------- |-------- |
| VideoMamba-Ti | K400    | 30    | 88            | 76.5           | 9.44     | [:hugs:HF](https://huggingface.co/Aristo2333/R-MeeTo/resolve/main/Table13/VideoMamba-Ti_30epoch_76.5.pth?download=true)| [🔗](https://drive.google.com/file/d/1oe0TPW9I-hWZs2Z0czgC9ggFbxsiya41/view?usp=drive_link)|
| VideoMamba-S  | K400    | 15    | 88            | 78.5           | 30.99    | [:hugs:HF](https://huggingface.co/Aristo2333/R-MeeTo/resolve/main/Table13/VideoMamba-S_15epoch_78.5.pth?download=true)| [🔗](https://drive.google.com/file/d/1Pr2Vo6W4q-rODG42zL_dq3koqW3JHNl5/view?usp=drive_link)|
| VideoMamba-B  | K400    | 15    | 88            | 78.9           | 67.11    | [🔗](https://drive.google.com/file/d/18A-w5vUfO70btqTsgyZtGhHQ-DrOfk5h/view?usp=sharing)| [🔗](https://drive.google.com/file/d/18UrCdukjZ9QqARFOppAujQLc75JWuHlT/view?usp=sharing)|
