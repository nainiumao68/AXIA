import os
import cv2
import torch
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._num_classes = setting.get('num_classes', None)
        self._ignore_index = setting.get('ignore_index', 255)
        self._label_mapping = setting.get('label_mapping', None)
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._gt_color_palette = setting.get('gt_color_palette', None)
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            if not hasattr(self, '_cached_expanded_filenames') or len(self._cached_expanded_filenames) != self._file_length:
                self._cached_expanded_filenames = self._construct_new_file_names(self._file_length)
            item_name = self._cached_expanded_filenames[index]
        else:
            item_name = self._file_names[index]
        rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
        x_path = os.path.join(self._x_path, item_name + self._x_format)
        gt_path = os.path.join(self._gt_path, item_name + self._gt_format)

        rgb = self._open_image(rgb_path, cv2.IMREAD_COLOR)
        if self._gt_color_palette is not None:
            gt = self._color_palette_to_index(gt_path, self._gt_color_palette)
        else:
            gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._label_mapping is not None:
            gt = self._apply_label_mapping(gt, self._label_mapping)
        if self._transform_gt:
            gt = self._gt_transform(gt)

        if self._x_single_channel:
            x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
            x = cv2.merge([x, x, x])
        else:
            x = self._open_image(x_path, cv2.IMREAD_COLOR)

        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))
        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        with open(source) as f:
            return [item.strip() for item in f.readlines()]

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_file_names += [self._file_names[i] for i in rand_indices[:length % files_len]]

        return new_file_names

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        if mode == cv2.IMREAD_GRAYSCALE:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        else:
            bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {filepath}")
        if dtype is not None:
            img = np.array(img, dtype=dtype)
        return img

    @staticmethod
    def _color_palette_to_index(filepath, palette):
        bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Label image not found or unreadable: {filepath}")
        gt = np.full(bgr.shape[:2], 255, dtype=np.uint8)
        for cls_idx, bgr_color in enumerate(palette):
            gt[np.all(bgr == np.array(bgr_color, dtype=np.uint8), axis=2)] = cls_idx
        return gt

    def _gt_transform(self, gt):
        transformed_gt = gt.astype(np.int32, copy=False)
        ignore_index = self._ignore_index if self._ignore_index is not None else 255
        num_classes = self._num_classes

        if num_classes is not None:
            if transformed_gt.min() >= 1 and transformed_gt.max() == num_classes:
                transformed_gt = transformed_gt - 1
                transformed_gt[transformed_gt < 0] = ignore_index
            transformed_gt[(transformed_gt < 0) & (transformed_gt != ignore_index)] = ignore_index
            transformed_gt[(transformed_gt >= num_classes) & (transformed_gt != ignore_index)] = ignore_index

        return transformed_gt.astype(np.uint8, copy=False)

    @staticmethod
    def _apply_label_mapping(gt, mapping):
        pairs = mapping.items() if isinstance(mapping, dict) else mapping
        arr = gt.astype(np.int32, copy=False) if gt.dtype not in (np.int32, np.int64) else gt
        lut = np.arange(256, dtype=np.int32)
        for src_val, dst_val in pairs:
            if 0 <= int(src_val) < 256:
                lut[int(src_val)] = int(dst_val)
        return lut[arr].astype(np.uint8, copy=False)
