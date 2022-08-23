#!/usr/bin/python
# -*- encoding: utf-8 -*-


import random
from PIL import Image
import PIL.ImageEnhance as ImageEnhance


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im, lb):
        if random.random() > self.p:
            return im, lb
        else:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            lb = lb.transpose(Image.FLIP_LEFT_RIGHT)
            return im, lb


class Rotation(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im, lb):
        if random.random() > self.p:
            return im, lb
        else:
            im = im.transpose(Image.ROTATE_90)
            lb = lb.transpose(Image.ROTATE_90)
            return im, lb


class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im, lb):
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        im = im.resize((w, h), Image.BILINEAR)
        lb = lb.resize((w, h), Image.NEAREST)
        return im, lb


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im, lb):
        for comp in self.do_list:
            im, lb = comp(im, lb)
        return im, lb

