#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
post process logMel spectrogram 
'''

import os
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        # self.inplace = inplace

    def __call__(self, imgs) :
        return (imgs - self.mean) / self.std


class SpecRandomCrop(object):
    """Spectrogram Random Crop

    Args:
        target_len (int): Length of target length, 
            only when input_length > target_length it will work.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, target_len=960, p=1.):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0, 1], got {p} instead.'

        self.target_len = target_len
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img

        cnt_len = img.shape[-1]

        if cnt_len > self.target_len:
            ## TODO

            if eval(os.environ.get("doing_eval", 'False')):
                crop_start = 0
                # print('### evaling crop ####')
            else:
                crop_start = random.randint(0, cnt_len - self.target_len)

            img = img[..., crop_start: crop_start + self.target_len]

        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'target_length = {self.target_len}, '
        repr_str += f'prob = {self.prob}'
        return repr_str


class SpecPadding(object):
    """Spectrogram Padding

    Args:
        target_len (int): Length of target length, tran
            if input_length < target_length, it will pad to target len;
            if input_length > target_length
    """

    def __init__(self, target_len=960, padding_method="circular"):

        self.target_len = target_len
        self.padding_method = padding_method

    def __call__(self, img):
        # if type(img) is not torch.Tensor:
        #     img = torch.tensor(img)
        img = self.padding_spec(img)

        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'target_length = {self.target_len}, '
        repr_str += f'padding_method = {self.padding_method}'
        return repr_str

    def padding_spec(self, img):
        """
        padding 频谱，提供了两种方式
        """
        
        cnt_len = img.shape[-1]
        if cnt_len > self.target_len:
            img = img[..., :self.target_len]
            attn_mask = np.ones_like(img) if 'zero' in self.padding_method else None
            return img,attn_mask

        padding_len = self.target_len - cnt_len

        out_tensor = np.zeros((1, img.shape[-2], self.target_len), dtype=img.dtype)
        attn_mask = np.zeros_like(out_tensor) if 'zero' in self.padding_method else None
        attn_mask_ones = np.ones_like(img)
        if self.padding_method == 'circular':
            # print("cnt_img:", cnt_img.shape)
            # out_tensor[idx] = F.pad(cnt_img.unsqueeze(0), (pad_before,pad_after,0,0), "circular")[0]

            repeat_times = self.target_len // cnt_len + 1
            if repeat_times % 2 == 0:
                repeat_times += 1

            cnt_img = np.tile(img, (1, 1, repeat_times))

            # print(f"cnt_img_repeat {repeat_times}:", cnt_img.shape)

            side_pad_num = cnt_img.shape[-1] // repeat_times * (repeat_times // 2)
            erase_side_pad_num = (cnt_img.shape[-1] - self.target_len + 1) // 2

            # print(side_pad_num, erase_side_pad_num)
            out_tensor[..., :] = cnt_img[..., erase_side_pad_num:erase_side_pad_num + self.target_len]

            # print("result:", out_tensor[idx].shape)

            pad_before = side_pad_num - erase_side_pad_num
            pad_after = padding_len - pad_before

            # print("pad:",pad_before, pad_after)
            # pad_lens_list.append((pad_before / float(target_len), pad_after / float(target_len)))

        elif self.padding_method == "random_circular":

            repeat_times = self.target_len // cnt_len + 3

            cnt_img = img.repeat((1, 1, repeat_times))

            # print(f"cnt_img_repeat {repeat_times}:", cnt_img.shape)


            # side_pad_num = cnt_img.shape[-1] // repeat_times * (repeat_times // 2)
            # erase_side_pad_num = (cnt_img.shape[-1] - self.target_len + 1) // 2
            choice_before = random.randint(0, cnt_img.shape[-1] - self.target_len)

            # print(side_pad_num, erase_side_pad_num)

            out_tensor[..., :] = cnt_img[..., choice_before:choice_before + self.target_len]

            # print("result:", out_tensor[idx].shape)

            # pad_before = side_pad_num - erase_side_pad_num
            # pad_after = padding_len - pad_before

            # print("pad:",pad_before, pad_after)
            # pad_lens_list.append((pad_before / float(target_len), pad_after / float(target_len)))

        elif self.padding_method == 'zero_before':
            pad_before = padding_len // 2
            pad_after = padding_len - pad_before
            out_tensor[..., pad_before:pad_before + cnt_len] = img
            attn_mask[..., pad_before:pad_before + cnt_len] = attn_mask_ones
            # pad_lens_list.append((pad_before / float(target_len), pad_after / float(target_len)))
        elif self.padding_method == "random_zero":
            pad_before = random.randint(0, padding_len // 2)
            pad_after = padding_len - pad_before
            out_tensor[..., pad_before:pad_before + cnt_len] = img
            attn_mask[..., pad_before:pad_before + cnt_len] = attn_mask_ones

        elif self.padding_method == "zero":
            # audio + zero padding 
            out_tensor[..., :cnt_len] = img
            attn_mask[..., :cnt_len] = attn_mask_ones 
        else:
            assert False, "padding method should belong to ('circular', 'zero')"

        return out_tensor#, attn_mask

class SpecMeanCrop(object):
    """
    substitution to SpecRamdonCrop 
    divide spectrogram into target length segments and get the average value
    work only when target_len > spect_len
    """
    def __init__(self, target_len=960, padding_method="circular"):
        self.target_len = target_len
        self.padding_method = padding_method 

    def __call__(self,img):
        cnt_len = img.shape[-1]

        if cnt_len > self.target_len:
            sub_imgs = [ img[..., idx:idx + self.target_len] for idx in range(0,cnt_len, self.target_len)]
            sub_imgs = sub_imgs[:-1] # since the last slice is unintact, we discard it
            sub_imgs = np.concat(sub_imgs)
            img = sub_imgs.mean(dim=0, keepdim=True)

        return img

class SpecRepeat(object):
    """Repeat channel layer
    """

    def __init__(self):
        pass

    def __call__(self, img):
        # if type(img) is tuple:
        #     attn_mask = img[1]
        #     img = img[0]
        #     if img.shape[0] == 1:
        #         img = img.repeat(3,1,1)
            
        #     if attn_mask is not None and attn_mask.shape[0] == 1:
        #         attn_mask = attn_mask.repeat(3,1,1)
        #         return img,attn_mask
        #     elif attn_mask is None:
        #         return img
        if img.shape[0] == 1:
            img = np.tile( img, (3,1,1))
        return img

def get_spectrogram(target_path):
        try:
            data = np.load(target_path)
            if data.dtype == np.half:
                data = data.astype(np.float32)
        except FileNotFoundError:
            print(f'load np logMel failed when loading {target_path}')
            return None

        assert data.shape[0] == 1 and data.ndim == 3, f"Data {target_path} shape is {data.shape}, corrupted!"

        return data
 
