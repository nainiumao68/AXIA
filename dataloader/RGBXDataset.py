"""Generic RGB-X semantic segmentation dataset.

Reads (RGB, X, label) triplets listed in a split file.  ``_open_image`` with
``IMREAD_COLOR`` returns RGB uint8 HWC (the internal BGR->RGB conversion
matches ImageNet-pretrained backbones); single-channel X modalities are read
as grayscale and replicated to 3 channels.
"""
import os

import cv2
import numpy as np
import torch
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
        # Optional label remapping, e.g. mapping rare classes to ignore or
        # compacting class indices.  Dict form: {raw_label: new_label}.
        self._label_mapping = setting.get('label_mapping', None)
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._rgb_four_channel = setting.get('rgb_is_four_channel', False)
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._dataset_name = setting.get('dataset_name', '').upper()
        # Color-palette labels: list of BGR tuples whose index is the class id;
        # None means the label map is already grayscale.
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

        if self._rgb_four_channel:
            # 4-channel RGBI input: keep all 4 channels, reordered BGRA -> RGBI.
            img_raw = self._open_image(rgb_path, cv2.IMREAD_UNCHANGED)
            if img_raw is not None and img_raw.ndim == 3 and img_raw.shape[2] == 4:
                rgb = img_raw[:, :, [2, 1, 0, 3]]
            else:
                rgb = self._open_image(rgb_path, cv2.IMREAD_COLOR)
        else:
            rgb = self._open_image(rgb_path, cv2.IMREAD_COLOR)
        if self._gt_color_palette is not None:
            gt = self._color_palette_to_index(gt_path, self._gt_color_palette)
        else:
            gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        # Label remapping (if configured) happens before the train/val preprocess.
        if self._label_mapping is not None and isinstance(self._label_mapping, dict):
            gt = self._apply_label_mapping(gt, self._label_mapping)
        elif self._label_mapping is not None and isinstance(self._label_mapping, (list, tuple)):
            gt = self._apply_label_mapping_pairs(gt, self._label_mapping)
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

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        """Unified image reading entry point.

        - ``IMREAD_COLOR`` (default): returns 3-channel RGB uint8; the internal
          BGR->RGB conversion matches the channel order expected by
          ImageNet-pretrained weights.
        - ``IMREAD_GRAYSCALE``: returns a single-channel grayscale image.
        - ``IMREAD_UNCHANGED``: returns the raw file content; the caller is
          responsible for the channel order.
        """
        if mode == cv2.IMREAD_GRAYSCALE:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        elif mode == cv2.IMREAD_UNCHANGED:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
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
        """Convert a BGR color label map to integer class indices via the
        palette; unmatched pixels become 255 (ignore)."""
        bgr = np.array(cv2.imread(filepath, cv2.IMREAD_COLOR), dtype=np.uint8)
        gt = np.full(bgr.shape[:2], 255, dtype=np.uint8)
        for cls_idx, bgr_color in enumerate(palette):
            mask = np.all(bgr == np.array(bgr_color, dtype=np.uint8), axis=2)
            gt[mask] = cls_idx
        return gt

    def _gt_transform(self, gt):
        """Robust label transform: auto-detects 1..K encodings and shifts them
        to 0..K-1; out-of-range values become the ignore index."""
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
        """LUT-based label remapping (dict form); order-independent."""
        arr = gt.astype(np.int32, copy=False) if gt.dtype not in (np.int32, np.int64) else gt
        lut = np.arange(256, dtype=np.int32)
        for src_val, dst_val in mapping.items():
            s = int(src_val)
            if 0 <= s < 256:
                lut[s] = int(dst_val)
        return lut[arr].astype(np.uint8, copy=False)

    @staticmethod
    def _apply_label_mapping_pairs(gt, pairs):
        """LUT-based label remapping (list-of-tuples form)."""
        arr = gt.astype(np.int32, copy=False) if gt.dtype not in (np.int32, np.int64) else gt
        lut = np.arange(256, dtype=np.int32)
        for src_val, dst_val in pairs:
            s = int(src_val)
            if 0 <= s < 256:
                lut[s] = int(dst_val)
        return lut[arr].astype(np.uint8, copy=False)
