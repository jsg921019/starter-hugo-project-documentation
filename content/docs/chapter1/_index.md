---
# Title, summary, and page position.
linktitle: Chapter 1
summary: Learn how to use Wowchemy's docs layout for publishing online courses, software documentation, and tutorials.
weight: 1
icon: book
icon_pack: fas

# Page metadata.
title: Chapter 1
date: "2018-09-09T00:00:00Z"
type: book  # Do not modify.
---

## Flexibility

Document everything!

This feature can be used for publishing content such as:

* **Online courses**
* **Project or software documentation**
* **Tutorials**
* **Notes**

The `courses` folder may be renamed. For example, we can rename it to `docs` for software/project documentation or `tutorials` for creating an online course.

## Delete courses

**To remove these pages, delete the `courses` folder and see below to delete the associated menu link.**

## Update site menu

After renaming or deleting the `courses` folder, you may wish to update any `[[main]]` menu links to it by editing your menu configuration at `config/_default/menus.toml`.

For example, if you delete this folder, you can remove the following from your menu configuration:

```toml
[[main]]
  name = "Courses"
  url = "courses/"
  weight = 50
```

Or, if you are creating a software documentation site, you can rename the `courses` folder to `docs` and update the associated *Courses* menu configuration to:

```toml
[[main]]
  name = "Docs"
  url = "docs/"
  weight = 50
```

## Update the docs menu

If you use the *docs* layout, note that the name of the menu in the front matter should be in the form `[menu.X]` where `X` is the folder name. Hence, if you rename the `courses/example/` folder, you should also rename the menu definitions in the front matter of files within `courses/example/` from `[menu.example]` to `[menu.<NewFolderName>]`.

## AlexNet

* 첫 성공적 모델

* https://arxiv.org/abs/1404.5997 버전이 여러개 있는듯 https://github.com/pytorch/vision/pull/704
* BUG : https://github.com/pytorch/vision/issues/549 : Dropout 순서가 잘못된듯
참조 : https://github.com/akrizhevsky/cuda-convnet2/blob/master/layers/layers-imagenet-1gpu.cfg

<img src='img/alexnet.png'/>


```python
import torch
import torch.nn as nn
import torchinfo
```


```python
class AlexNet(nn.Module):
    
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # Convolution Layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4), # (227-11)/4 + 1 = 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (55-3)/2 + 1 = 27
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # (27+2*2-5) + 1 = 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (27-3)/2 + 1 = 13
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # (13+1*2-3) + 1 = 13
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # (13+1*2-3) + 1 = 13
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # (13+1*2-3) + 1 = 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (13-3)/2 + 1 = 6
        )
        
        # Linear Layers
        self.classifier = nn.Sequential(
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```
