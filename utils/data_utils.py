import logging
import torch.utils.data as data

import librosa
from PIL import Image
import os
import os.path
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.random import randint
import pickle
import torch
from .transforms import *
from torchvision import transforms

logger = logging.getLogger(__name__)


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L' or img_group[0].mode == 'F':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.to(torch.float32).div(255) if self.div else img.to(torch.float32)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def start_frame(self):
        return 0

    @property
    def untrimmed_video_name(self):
        return self._data[0]

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return {'RGB': int(self._data[1]),
                'Spec': int(self._data[1])}

    @property
    def label(self):
        return int(self._data[2])
        # target = self._data[2].strip().split(',')
        # for i in range(len(target)):
        #     target[i] = int(target[i])
        # return target


def RandomMaskingGenerator(num_patches, ratio):

    num_mask = int(num_patches * ratio)
    mask = np.hstack([
        np.zeros(num_patches - num_mask),
        np.ones(num_mask),
    ])
    np.random.shuffle(mask)
    return mask


class MultiDataSet(data.Dataset):
    def __init__(self, dataset, list_file,
                 new_length, modality, image_tmpl,
                 visual_path=None, audio_path=None,
                 resampling_rate=44000,
                 num_segments=3, transform=None,
                 mode='train', use_audio_dict=True,
                 mask_ratio=0.5, ):
        self.dataset = dataset
        if audio_path is not None:
            if not use_audio_dict:
                self.audio_path = Path(audio_path)
            else:
                self.audio_path = pickle.load(open(audio_path, 'rb'))
        self.visual_path = visual_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.mode = mode
        self.resampling_rate = resampling_rate
        self.use_audio_dict = use_audio_dict

        self.ratio = mask_ratio
        self.num_tokens = {'RGB': 1568, 'Spec': 512}

        if 'RGBDiff' in self.modality:
            self.new_length['RGBDiff'] += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _log_specgram(self, audio, window_size=10,
                      step_size=5, eps=1e-6):
        nperseg = int(round(window_size * self.resampling_rate / 1e3))
        noverlap = int(round(step_size * self.resampling_rate / 1e3))

        spec = librosa.stft(audio, n_fft=254,
                            window='hann',
                            hop_length=noverlap,
                            win_length=nperseg,
                            pad_mode='constant')
        spec = np.log(np.real(spec * np.conj(spec)) + eps)
        return spec

    def _extract_sound_feature(self, record, idx):
        centre_sec = idx / 25
        left_sec = centre_sec - 0.3195
        right_sec = centre_sec + 0.3195
        audio_fname = record.path + '.wav'
        if not self.use_audio_dict:
            samples, sr = librosa.core.load(self.audio_path / audio_fname, sr=24000, mono=True)
        else:
            audio_fname = record.untrimmed_video_name
            # audio_fname = audio_fname.split('/')[1]
            samples = self.audio_path[audio_fname]

        duration = samples.shape[0] / float(self.resampling_rate)

        left_sample = int(round(left_sec * self.resampling_rate))
        right_sample = int(round(right_sec * self.resampling_rate))

        if left_sec < 0:
            samples = samples[:int(round(self.resampling_rate * 0.639))]

        elif right_sec > duration:
            samples = samples[-int(round(self.resampling_rate * 0.639)):]
        else:
            samples = samples[left_sample:right_sample]

        return self._log_specgram(samples)

    def _load_data(self, modality, record, idx):
        if modality == 'RGB' or modality == 'RGBDiff':
            if idx == 0:
                idx_untrimmed = record.start_frame + idx + 1
            else:
                idx_untrimmed = record.start_frame + idx
            return [Image.open(os.path.join(self.visual_path, record.untrimmed_video_name,
                                            self.image_tmpl[modality].format(idx_untrimmed))).convert('RGB')]
        elif modality == 'Flow':
            rgb2flow_fps_ratio = record.fps['Flow'] / float(record.fps['RGB'])
            idx_untrimmed = int(np.floor((record.start_frame * rgb2flow_fps_ratio))) + idx
            x_img = Image.open(os.path.join(self.visual_path, record.untrimmed_video_name,
                                            self.image_tmpl[modality].format('x', idx_untrimmed))).convert('L')
            y_img = Image.open(os.path.join(self.visual_path, record.untrimmed_video_name,
                                            self.image_tmpl[modality].format('y', idx_untrimmed))).convert('L')
            return [x_img, y_img]
        elif modality == 'Spec':
            # audio_fname = record.path + '.wav'
            audio_fname = record.path + '.mp3'
            # if os.path.exists('/opt/data/private/datasets/UCF101/ucf101_audio/' + audio_fname):
            if os.path.exists('/opt/data/private/datasets/homage/audio/' + audio_fname):
                spec = self._extract_sound_feature(record, idx)
            else:
                print(audio_fname)
                spec = np.zeros((128, 128))
            return [Image.fromarray(spec)]

    def _parse_list(self):
        if self.dataset == 'epic-kitchens-55':
            self.video_list = [EpicKitchens55_VideoRecord(tup) for tup in self.list_file.iterrows()]
        elif self.dataset == 'epic-kitchens-100':
            self.video_list = [EpicKitchens100_VideoRecord(tup) for tup in self.list_file.iterrows()]
        else:
            # check the frame number is large >3:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]

            tmp = [item for item in tmp if int(item[1]) >= 3]
            self.video_list = [VideoRecord(item) for item in tmp]

            print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record, modality):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames[modality] - self.new_length[modality] + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record, modality):
        if record.num_frames[modality] > self.num_segments + self.new_length[modality] - 1:
            tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record, modality):

        tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):

        input = {}
        mask = {}
        record = self.video_list[index]

        for m in self.modality:
            if self.mode == 'train':
                segment_indices = self._sample_indices(record, m)
            elif self.mode == 'val':
                segment_indices = self._get_val_indices(record, m)
            elif self.mode == 'test':
                segment_indices = self._get_test_indices(record, m)

            # We implement a Temporal Binding Window (TBW) with size same as the action's length by:
            #   1. Selecting different random indices (timestamps) for each modality within segments
            #      (this is similar to using a TBW with size same as the segment's size)
            #   2. Shuffling randomly the segments of Flow, Audio (RGB is the anchor hence not shuffled)
            #      which binds data across segments, hence making the TBW same in size as the action.
            #   Example of an action with 90 frames across all modalities:
            #    1. Synchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [12, 41, 80], Audio: [12, 41, 80]
            #    2. Asynchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [9, 55, 88], Audio: [20, 33, 67]
            #    3. Asynchronous selection of indices per action:
            #       RGB: [12, 41, 80], Flow: [88, 55, 9], Audio: [67, 20, 33]

            # if m != 'RGB' and self.mode == 'train':
            #     np.random.shuffle(segment_indices)

            img, label = self.get(m, record, segment_indices)
            input[m] = img
            mask[m] = RandomMaskingGenerator(self.num_tokens[m], self.ratio)

        # label = label - 1
        return input, label, mask

    def get(self, modality, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length[modality]):
                seg_imgs = self._load_data(modality, record, p)
                images.extend(seg_imgs)
                if p < record.num_frames[modality]:
                    p += 1

        process_data = self.transform[modality](images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


def get_loader(args):
    image_tmpl = {}
    train_transform = {}
    val_transform = {}
    data_length = {}
    normalize = {}
    for m in args.modality:
        if m != 'Spec':
            if m != 'RGBDiff':
                normalize[m] = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            else:
                normalize[m] = IdentityTransform()

    for m in args.modality:
        data_length[m] = 1 if (m == "RGB" or m == "Spec") else 5

    for m in args.modality:
        if m != 'Spec':
            # Prepare dictionaries containing image name templates for each modality
            if m in ['RGB', 'RGBDiff']:
                image_tmpl[m] = "{:05d}.jpg"
            elif m == 'Flow':
                image_tmpl[m] = args.flow_prefix + "{}_{:010d}.jpg"
            # Prepare train/val dictionaries containing the transformations
            # (augmentation+normalization)
            # for each modality
            train_transform[m] = transforms.Compose([
                GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                GroupRandomHorizontalFlip(is_flow=False),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize[m],
            ])

            val_transform[m] = transforms.Compose([
                GroupScale(int(256)),
                GroupCenterCrop(224),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                normalize[m],
            ])
        else:
            # Prepare train/val dictionaries containing the transformations
            # (augmentation+normalization)
            # for each modality
            train_transform[m] = transforms.Compose([
                Stack(roll=False),
                ToTorchFormatTensor(div=False),
            ])

            val_transform[m] = transforms.Compose([
                Stack(roll=False),
                ToTorchFormatTensor(div=False),
            ])

    train_loader = torch.utils.data.DataLoader(
        MultiDataSet(args.dataset,
                     args.train_list,
                     data_length,
                     args.modality,
                     image_tmpl,
                     visual_path=args.visual_path,
                     audio_path=args.audio_path,
                     num_segments=args.num_segments,
                     transform=train_transform,
                     resampling_rate=args.resampling_rate),
        batch_size=args.train_batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        MultiDataSet(args.dataset,
                     args.val_list,
                     data_length,
                     args.modality,
                     image_tmpl,
                     visual_path=args.visual_path,
                     audio_path=args.audio_path,
                     num_segments=args.num_segments,
                     mode='val',
                     transform=val_transform,
                     resampling_rate=args.resampling_rate),
        batch_size=args.train_batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader


def mixup_data(x, y, alpha):
    # Returns mixed inputs, pairs of targets, and lambda
    mixed_x = {}
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x['RGB'].size()[0])
    for modality, values in x.items():
        mixed_x[modality] = lam * values + (1.0 - lam) * values[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
