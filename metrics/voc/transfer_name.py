import os
opj = os.path.join

VOC_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
TYPE = 'groundtruths'

for name in os.listdir(TYPE):
    label_path = opj(TYPE, name)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    for idx, label in enumerate(labels):
        temp_label = label.split(' ')
        temp_label[0] = VOC_NAMES[int(temp_label[0])]
        labels[idx] = ' '.join(temp_label)
    os.remove(label_path)
    with open(label_path, 'w') as f:
        f.writelines(labels)
