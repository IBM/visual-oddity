# Copyright (c) 2023 IBM Research. All Rights Reserved.
#
# Code accompanying a manuscript entitled:
# "On the visual analytic intelligence of neural networks"

from random import shuffle
import os
import random
import collections
import hashlib
from PIL import Image, ImageDraw
import sys
from generators import *
from constants import RIDDLE_IDS, NUM_IMAGES, NUM_RIDDLES

generators = {0: ColorGenerator, 1: OrientationGenerator, 2: LineGenerator, 3: LineGenerator, 4: PointOnLineGenerator,
              5: PointOnLineGenerator, 6: ParallelGenerator, 7: ParallelGenerator, 8: RightAngleGenerator, 9: RightAngleGenerator,
              10: QuadrilateralGenerator, 11: TrapezoidGenerator, 12: ParallelogramGenerator, 13: RectangleGenerator, 14: SquareGenerator,
              15: EquilateralGenerator, 16: RightAngleTriangleGenerator, 17: CircleGenerator, 18: ConvexGenerator, 19: SymmetricalGenerator,
              20: SymmetricalGenerator, 21: SymmetricalGenerator, 22: ChiralGenerator, 23: ChiralGenerator, 24: ChiralGenerator,
              25: ChiralGenerator, 26: DistanceGenerator, 27: PointDistanceGenerator, 28: PointDistanceGenerator, 29: CircleCenterGenerator,
              30: QuadrilateralCenterGenerator, 31: ProportionsGenerator, 32: ProportionsGenerator, 33: TranslationGenerator, 34: FixedGenerator,
              35: FixedGenerator, 36: SymmetryGenerator, 37: SymmetryGenerator, 38: SymmetryGenerator, 39: PointSymmetryGenerator,
              40: RotationGenerator, 41: InsideGenerator, 42: ClosureGenerator, 43: ConnectednessGenerator, 44: HolesGenerator,}

def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()

class Generator():
    def __init__(self, output_dir, riddle_set, bg_colors=list(range(235,256)), img_size=100):
        self.output_dir = output_dir; utils.create_dirs([output_dir])
        self.riddle_set = riddle_set
        self.bg_colors = bg_colors
        self.args_img_size = collections.namedtuple('args',['img_size','only_oddity'])(img_size, False)
        np.seterr(all='raise') #required to validate the generated points for riddle 41

    def generate(self, name='train', num_samples=1600):
        output_path = os.path.join(self.output_dir, name + '.tfrecords')
        writer = tf.python.python_io.TFRecordWriter(output_path)
        for i in range(num_samples):
            [sample, target_id, riddle_id] = self.generate_sample(i)
            feature = { 'label': _int64_feature(target_id),
                        'riddle_id': _int64_feature(riddle_id) }
            for idx, img in enumerate(sample):
                feature['img' + str(idx)] = _bytes_feature(img.tobytes())
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString(deterministic=True))
        writer.close()
        return output_path

    def generate_sample(self, id):
        riddle_set = self.riddle_set
        riddle_id = choice(riddle_set) #random riddle from a set of possible riddles
        generator = generators[riddle_id](RIDDLE_IDS[riddle_id], self.args_img_size, False, self.bg_colors)
        images = generator.get_images()
        context = images[:NUM_IMAGES - 1]
        target = images[NUM_IMAGES - 1]
        shuffle(context)
        target_id = randint(0, NUM_IMAGES - 1)
        sample = context[:target_id] + [target] + context[target_id:]
        return [sample, target_id, riddle_id]

    def generate_images(self, NUM=8):
        s = 100
        m = 8 #margin
        img = Image.new('RGB', (6*s + 7*m, NUM * (s+2*m)), color='white')
        d = ImageDraw.Draw(img)
        for b in range(NUM):
            [sample, target_id, riddle_id] = self.generate_sample(0)
            x = m
            y = m+b*(s+2*m) #row-wise
            for idx, image in enumerate(sample):
                img.paste(image, (x, y))
                outline = 'black'
                if idx == target_id:
                    outline = 'red'
                d.rectangle([x, y, x + s, y + s], outline=outline, width=4)
                x += s + m
        img.show()

def set_seed(seed_val):
    random.seed(seed_val) # equiv to: random._inst.seed(seed_val)
    np.random.seed(seed_val)

def files_md5(files):
    md_all = ''
    for f in files:
        md = str(md5sum(f))
        print(f + '  md5sum: ' + md)
        open(f.replace('tfrecords', md + '.md5'), 'a').close()
        md_all += md

####### GENERATE DATASETS ########
base = None
seed_set = list(range(10))
riddle_set = list(range(NUM_RIDDLES))
ds = 3200

for a in sys.argv:
    if '--base' in a:
        base = a.split('=')[1]
    if '--seed' in a:
        seed_set = [ int(a.split('=')[1]) ]
    if '--riddle' in a:
        riddle_set = [ int(a.split('=')[1]) ]
    if '--size' in a:
        ds = int(a.split('=')[1])

if base is None:  # If not output directory is given, just display examples:
    g = Generator(output_dir='.', riddle_set=[random.randint(0,NUM_RIDDLES)])
    g.generate_images()
    exit(0)

# TensorFlow is only needed for writing
import tensorflow as tf
print('Your TensorFlow version is', tf.__version__, '. Recommended: 1.15.0')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for seed_val in seed_set:
    print('Seed '+str(seed_val))
    files = []
    for id in riddle_set:
        if id < NUM_RIDDLES:
            print('Generation for riddle id='+str(id))
            g = Generator(output_dir=base+'riddles_'+str(ds)+'/s'+str(seed_val)+'/riddle'+str(id), riddle_set=[id])
            set_seed(id+100*(0)+seed_val); files.append(g.generate(name='valid', num_samples=ds//5))
            set_seed(id+100*(1)+seed_val); files.append(g.generate(name='train', num_samples=ds//5*4))
            set_seed(id+100*(2)+seed_val); files.append(g.generate(name='test', num_samples=ds//5))
    files_md5(files)
    print('(individual riddles)')

    if 90 in riddle_set:
        print('==========')
        print('Generation for all riddles')
        files = []
        g = Generator(output_dir=base+'riddles_90000/s'+str(seed_val)+'/riddle90k', riddle_set=list(range(NUM_RIDDLES)))
        set_seed(id+100*(0)+seed_val); files.append(g.generate(name='valid', num_samples=18000))
        set_seed(id+100*(1)+seed_val); files.append(g.generate(name='train', num_samples=72000))
        set_seed(id+100*(2)+seed_val); files.append(g.generate(name='test', num_samples=18000))
        files_md5(files)
        print('(all riddles)')
