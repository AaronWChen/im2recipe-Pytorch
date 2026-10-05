import os
import sys

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from recipe_store import RecipeStore


def default_loader(path):
    try:
        return Image.open(path).convert('RGB')
    except Exception as e:
        print('Could not load image %s (%s); using a white one instead.' % (path, e), file=sys.stderr)
        return Image.new('RGB', (224, 224), 'white')


class ImagerLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader, square=False, data_path=None, partition=None, sem_reg=None):

        if data_path is None:
            raise ValueError('No data path specified.')
        if partition not in ('train', 'val', 'test'):
            raise ValueError('Unknown partition type %s.' % partition)
        self.partition = partition

        # <data_path>/<partition>_store/ is written by scripts/build_dataset.py
        self.store = RecipeStore(os.path.join(data_path, partition + '_store'))

        self.square = square
        self.imgPath = img_path
        self.mismtch = 0.8
        self.maxInst = 20

        self.semantic_reg = bool(sem_reg) if sem_reg is not None else False

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def _image_path(self, index):
        names = self.store.image_names(index)
        if self.partition == 'train':
            # We do only use the first five images per recipe during training
            img_idx = np.random.choice(range(min(5, len(names))))
        else:
            img_idx = 0
        name = names[img_idx]
        # images live in a four-level folder structure, e.g. 0fa8309c13.jpg -> 0/f/a/8/0fa8309c13.jpg
        return os.path.join(self.imgPath, *name[:4], name)

    def __getitem__(self, index):
        store = self.store
        # we force 80 percent of them to be a mismatch
        if self.partition == 'train':
            match = np.random.uniform() > self.mismtch
        else:
            match = True
        target = 1 if match else -1

        if target == 1:
            img_index = index
        else:
            # we randomly pick one non-matching image
            all_idx = range(len(store))
            img_index = np.random.choice(all_idx)
            while img_index == index:
                img_index = np.random.choice(all_idx)
        path = self._image_path(img_index)

        # instructions
        instrs = store.instructions(index)
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, instrs.shape[1]), dtype=np.float32)
        t_inst[:itr_ln] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        raw_ingrs = store.ingredients(index)
        ingrs = torch.LongTensor(raw_ingrs.astype(int))
        igr_ln = max(np.nonzero(raw_ingrs)[0]) + 1

        # load image
        img = self.loader(path)

        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        rec_class = store.recipe_class(index)
        rec_id = store.recipe_id(index)
        img_class = store.recipe_class(img_index)
        img_id = store.recipe_id(img_index)

        # output
        if self.partition == 'train':
            if self.semantic_reg:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_class, rec_class]
            else:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target]
        else:
            if self.semantic_reg:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_class, rec_class, img_id, rec_id]
            else:
                return [img, instrs, itr_ln, ingrs, igr_ln], [target, img_id, rec_id]

    def __len__(self):
        return len(self.store)
