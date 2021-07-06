import json
import shutil
from tqdm import tqdm
import os
from PIL import Image

with open('ChestRayNIH-train.json','r')as f:
    train = json.load(f)

for img in tqdm(train['annotations']):
    dst = img['path']
    src = os.path.join('/home/432/qihaoyu/data/ChestRayNIH/images/',dst.split('/')[-1])
    img['path']=dst
    for i in range(10):
        shutil.copy(src,dst)
        try:
            Image.open(dst)
            break
        except:
            continue

with open('ChestRayNIH-test.json','r')as f:
    test = json.load(f)
for img in tqdm(test['annotations']):
    dst = img['path']
    src = os.path.join('/home/432/qihaoyu/data/ChestRayNIH/images/',dst.split('/')[-1])
    img['path']=dst
    for i in range(10):
        shutil.copy(src,dst)
        try:
            Image.open(dst)
            break
        except:
            continue

