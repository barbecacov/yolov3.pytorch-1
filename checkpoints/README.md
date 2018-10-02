# Checkpoints

Folder should be organized like

```
.
├── 0.ckpt
├── coco/
│   └── -1.ckpt
├── darknet/
│   ├── darknet53.conv.74.weights
│   └── yolov3-coco.weights
└── tejani/
```

Sub folder name corresponds to dataset. Each folder contains checkpoints file, each is named with saved epoch.

Special files:

* `-1.ckpt`, official pre-trained model
* `0.ckpt`, checkpoints right after loading pre-trained backbone parameters
* `*.weights`, darknet format checkpoint
