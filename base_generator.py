# Copyright (c) 2023 IBM Research. All Rights Reserved.
#
# Code accompanying a manuscript entitled:
# "On the visual analytic intelligence of neural networks"

from random import randint, choice
from PIL import Image, ImageDraw
from constants import NUM_IMAGES, MAX_COLOR_INTENSITY

class BaseGenerator():
    def __init__(self, name, args, test=False, bg_colors=None):
        self.args = args
        self.img_size = args.img_size
        self.color = self.get_random_color()
        self.images = []
        self.name = name
        self.image_type = 'L'
        self.test = test
        self.bg_colors = bg_colors  #provide a list of possible bg_colors to avoid training/test fixed colors

    def get_images(self):
        self.set_global_state()
        for i in range(6):
            self.img, d = self.create_image()
            self.is_oddity = i == NUM_IMAGES - 1
            if self.args.only_oddity:
                self.is_oddity = True
            success = False
            while not success:  # in some cases numerical errors may arise, so retry
                try:
                    self.set_state()
                    success = True
                except FloatingPointError:
                    success = False
            if not self.is_oddity:
                self.generate(d)
            else:
                self.generate_oddity(d)
            self.images.append(self.img)
        return self.images

    def create_image(self):
        if self.bg_colors is not None:
            color = choice(self.bg_colors)
        else:
            color = 255 if self.test else randint(235, 254)
        img = Image.new(self.image_type, (self.img_size, self.img_size), color=color)
        d = ImageDraw.Draw(img)
        return img, d

    def get_random_color(self):
        return randint(0, MAX_COLOR_INTENSITY)

    def set_global_state(self):
        pass

    def set_state(self):
        pass

    def generate(self, d):
        raise NotImplementedError

    def generate_oddity(self, d):
        raise NotImplementedError
