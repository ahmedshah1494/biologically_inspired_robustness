import os
import torch
import torchvision
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--idir', required=True)
parser.add_argument('--odir', required=True)
args = parser.parse_args()

T = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
])
dataset = torchvision.datasets.ImageFolder(args.idir, transform=T)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=lambda x: [x_[0] for x_ in x])
classes = [
    2,# 'car',
    15,# 'cat',
    60,# 'dining table',
    26,# 'handbag',
    17# 'horse',
]
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes=classes
# cls2idx = {v:k for k,v in model.names.items()}
# class_idxs = [cls2idx[c] for c in classes]
bboxes = []
for imgs in tqdm(loader):
    # Inference
    results = model(imgs)
    bboxes.extend(results.xyxyn)

for bb, (fp,_) in tqdm(zip(bboxes, dataset.samples)):
    ofp = os.path.join(args.odir, *(fp.split('/')[-3:]))
    if len(bb) > 0:
        bb = [b.cpu().numpy() for b in bb]
        if not os.path.exists(os.path.dirname(ofp)):
            os.makedirs(os.path.dirname(ofp))
        np.savetxt(ofp, bb)


# # Results
# results.print()
# results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]