## Visual Oddity

This repository contains code material for the publication: 
S. Woźniak, H. Jónsson, G. Cherubini, A. Pantazi, and E. Eleftheriou, ‘On the visual analytic intelligence of neural networks’.

In particular, the code implements programmatic generation of a dataset for the visual oddity task. The task was first proposed in S. Dehaene, V. Izard, P. Pica, and E. Spelke, ‘Core Knowledge of Geometry in an Amazonian Indigene Group’, Science, vol. 311, no. 5759, pp. 381–384, Jan. 2006, doi: 10.1126/science.1121739 . It is defined on geometrical objects and consists of 45 distinct riddles designed to discover which basic concepts of geometry such as points, lines, parallelism, and symmetry, are understood by the humans. Each riddle contains six frames, five of which include a geometrical concept being tested. One of the frames violates the geometrical concept and is called the oddity. The goal is to classify which one of the six frames is the oddity. 

## Usage

### Dependencies

Installation of the dependencies typically takes a few minutes. The steps below were tested on a laptop running Windows 11 operating system, but should be similar for other systems.

An environment with required packages can be created by running:
`conda create -n visual-oddity python=3.6.12 pillow=7.1.2 shapely=1.7.1 scipy=1.5.2 numpy=1.16.6 tensorflow=1.15`
where `tensorflow` or `tensorflow-gpu` (if GPU is present on the PC) package is 
**only required for generating the output files - can be skipped for visualization only.**

Please activate the environment and execute the code:
```
conda activate visual-oddity
python main.py
```

### Visualization of examples
Visualization functionality displays on the screen 8 examples for a randomly selected riddle.
The program should execute instantaneously. 

`main.py` - displays examples for random riddle
`main.py --riddle=12` - displays examples for a given riddle. See riddle IDs below:

```
generators = {0: ColorGenerator, 1: OrientationGenerator, 2: LineGenerator, 3: LineGenerator, 4: PointOnLineGenerator,
              5: PointOnLineGenerator, 6: ParallelGenerator, 7: ParallelGenerator, 8: RightAngleGenerator, 9: RightAngleGenerator,
              10: QuadrilateralGenerator, 11: TrapezoidGenerator, 12: ParallelogramGenerator, 13: RectangleGenerator, 14: SquareGenerator,
              15: EquilateralGenerator, 16: RightAngleTriangleGenerator, 17: CircleGenerator, 18: ConvexGenerator, 19: SymmetricalGenerator,
              20: SymmetricalGenerator, 21: SymmetricalGenerator, 22: ChiralGenerator, 23: ChiralGenerator, 24: ChiralGenerator,
              25: ChiralGenerator, 26: DistanceGenerator, 27: PointDistanceGenerator, 28: PointDistanceGenerator, 29: CircleCenterGenerator,
              30: QuadrilateralCenterGenerator, 31: ProportionsGenerator, 32: ProportionsGenerator, 33: TranslationGenerator, 34: FixedGenerator,
              35: FixedGenerator, 36: SymmetryGenerator, 37: SymmetryGenerator, 38: SymmetryGenerator, 39: PointSymmetryGenerator,
              40: RotationGenerator, 41: InsideGenerator, 42: ClosureGenerator, 43: ConnectednessGenerator, 44: HolesGenerator,}
```

### Generation of per-riddle datasets

Generation functionality creates `.tfrecords` dataset files.
With default settings, the execution time on a laptop is approx. 3 hours.

`python main.py --base=./output_directory/` - specify an output directory to generate datasets

**Parameters:**
* `--seed=0` - select one specified random seed for generation. Default: generate for ten seeds: 0,1,...,9
* `--riddle=12` - select one specified riddle ID for generation. Default: generate for all individual riddles: 0,1,...,44
* `--size=3200` - number of examples per riddle and seed seen during training. Default: 3200. 
**Note:** Training examples are split with 80/20 ratio between training and validation files, leading to 4/5\*X training and 1/5\*X validation examples. Additional test set of size 1/X is generated. Thus, in total 6/5\*X examples are generated.  


### Generation of combined dataset

Generation functionality creates `.tfrecords` dataset files.
With default settings, the execution time on a laptop is approx. 2 hours.

`python main.py --base=./output_directory/ --riddle=90` - specifying a "special" riddle ID of 90 generates a combined dataset with 90000 examples for the training phase. 

**Parameters:**
* `--seed=0` - select one specified random seed for generation. Default: generate for ten seeds: 0,1,...,9

