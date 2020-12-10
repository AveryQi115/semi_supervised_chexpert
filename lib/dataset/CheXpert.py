from dataset.baseset import BaseSet
import torchvision.transforms as transforms
from data_transform.transform_wrapper import TRANSFORMS
import random, cv2
import os
import numpy as np
import ipdb

class CheXpert(BaseSet):
    def __init__(self, mode='train', cfg=None, transform=None):
        super().__init__(mode, cfg, transform)
        random.seed(0)

        transform_uncertain = [{"nan":0,0:0,1:0,-1:0},
                                {"nan":0,0:0,1:0,-1:1},
                                {"nan":0,0:0,1:0,-1:-1}]
        if cfg.DATASET.UNCERTAIN == "U-positive":
            self.transform_dict = transform_uncertain[1]
        elif cfg.DATASET.UNCERTAIN == "U-negative":
            self.transform_dict == transform_uncertain[0]
        elif cfg.DATASET.UNCERTAIN == "U-ignore":
            self.transform_dict == transform_uncertain[2]

        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and mode=="train"):
            self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        now_info = self.data[index]
        img = self._get_image(now_info)
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img,(3,3),0)
        image = self.transform(img)
        assert image.shape == (1,320,320), f'image.shape={image.shape}'
        label = self._get_label(now_info)
        # meta = self._get_meta(now_info)
        meta = dict()
        return image, label, meta

    def _get_image(self,now_info):
        if self.data_type == "jpg":
            fpath = os.path.join(self.data_root, now_info["path"])
            img = cv2.imread(fpath,0)
        elif self.data_type == "png":
            fpath = os.path.join(self.data_root, now_info["path"])
            img = cv2.imread(fpath,0)
        # assert img.shape == (320,320), f'img.shape={img.shape}'
        return img

    def _get_meta(self,now_info):
        meta = dict()
        for key in now_info.keys():
            if key=="path":
                continue
            else:
                meta[key] = now_info[key]
        return meta

    def _get_label(self,now_info):
        label = []
        for key in now_info.keys():
            if key in ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']:
                label.append(self.transform_dict[now_info[key]])
        label = np.array(label)
        assert label.shape == (14,)
        return label

    def update_transform(self, input_size=None):
        transform_list = [transforms.ToPILImage()]
        transform_ops = (
            self.cfg.TRANSFORMS.TRAIN_TRANSFORMS
            if self.mode == "train"
            else self.cfg.TRANSFORMS.TEST_TRANSFORMS
        )
        for tran in transform_ops:
            transform_list.append(TRANSFORMS[tran](cfg=self.cfg, input_size=self.input_size))
        transform_list.extend([transforms.ToTensor()])
        # print(transform_list)
        self.transform = transforms.Compose(transform_list)

    def get_annotations(self):
        clean_anno = []
        for anno in self.data:
            clean_anno_dict=dict()
            for key in anno.keys():
                if key not in ['path','Sex','Age','Frontal/Lateral','AP/PA']:
                    clean_anno_dict[key] = self.transform_dict[anno[key]]
            clean_anno.append(clean_anno_dict)
        return clean_anno