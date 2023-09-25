# Copyright (c) 2023 IBM Research. All Rights Reserved.
#
# Code accompanying a manuscript entitled:
# "On the visual analytic intelligence of neural networks"

import os
import math
from random import randint, uniform
import numpy as np
from geometry import Line, Point

def create_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def random_point(img_size, offset=0.1):
    min_value = int(offset * img_size)
    max_value = int(img_size) - min_value
    return (randint(min_value, max_value), randint(min_value, max_value))

def random_points(num_points, img_size, offset=0.1):
    pnts = []
    while len(pnts) < num_points:
        pnt = random_point(img_size, offset)
        if pnt not in pnts:
            pnts.append(pnt)
    return np.array(pnts)

def rotate(origin, point, angle): #angle in radians
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def get_random_rotation():
    return math.radians(randint(0, 180))

def get_random_multiplier():
    return 1 if randint(0, 1) else -1

def get_random_of_percentage(min_percentage, max_percentage, img_size):
    min_value = min_percentage * img_size
    max_value = max_percentage * img_size
    return randint(int(min_value), int(max_value))

def get_boundary_points(pts):
    min_x = np.amin(np.array(pts)[:, 0])
    min_y = np.amin(np.array(pts)[:, 1])
    max_x = np.amax(np.array(pts)[:, 0])
    max_y = np.amax(np.array(pts)[:, 1])
    return min_x, min_y, max_x, max_y

def get_bbox(pts):
    pts = np.array(pts)
    min_x = min(pts[:, 0])
    min_y = min(pts[:, 1])
    max_x = max(pts[:, 0])
    max_y = max(pts[:, 1])
    return min_x, min_y, max_x, max_y

def get_bbox_lengths(pts):
    min_x, min_y, max_x, max_y = get_bbox(pts)
    return max_x - min_x, max_y - min_y

def get_minmax_offsets(pts, img_size):
    min_x, min_y, max_x, max_y = get_boundary_points(pts)
    max_x_offset = img_size * 0.9 - max_x
    min_x_offset = min_x - img_size * 0.1
    max_y_offset = img_size * 0.9 - max_y
    min_y_offset = min_y - img_size * 0.1
    return min_x_offset, min_y_offset, max_x_offset, max_y_offset

def get_random_rectangle_props(args, offset=0.1, as_dict=False, square=False, max_size=0.5):
    width = get_random_of_percentage(0.2, max_size, args.img_size)
    height = get_random_of_percentage(0.2, max_size, args.img_size)
    while abs(width - height) < 0.05 * args.img_size:
        height = get_random_of_percentage(0.2, max_size, args.img_size)  # dont confuse with the square test
    if square:
        height = width
    diameter = math.sqrt(math.pow(width // 2, 2) + math.pow(height // 2, 2))
    min_value = offset * args.img_size + diameter
    max_value = (1 - offset) * args.img_size - diameter
    x = uniform(min_value, max_value)
    y = uniform(min_value, max_value)
    assert x - diameter >= offset * args.img_size and y - diameter >= offset * args.img_size
    assert x + diameter <= args.img_size * (1 - offset) and y + diameter <= (1 - offset) * args.img_size
    x = x - width // 2
    y = y - height // 2
    if not as_dict:
        return x, y, width, height
    else:
        return {'x': x, 'y': y, 'width': width, 'height': height}

def get_random_line_props(args, offset=0.0):
    length = get_random_of_percentage(0.4, 0.6 - offset, args.img_size)
    x = get_random_of_percentage(0.35 + offset, 0.65 - offset, args.img_size)
    y = get_random_of_percentage(0.35 + offset, 0.65 - offset, args.img_size)
    width = get_random_of_percentage(0.01, 0.03, args.img_size)
    rotation = get_random_rotation()
    pts = [(x - length // 2, y), (x + length // 2, y)]
    props = {'length': length, 'x': x, 'y': y, 'width': width, 'center': (x, y), 'rotation': rotation, 'pts': pts}
    return props

def translate_points(pts, offset, axis=None):
    pts = np.array(pts)
    if axis is None:
        return pts + np.array(offset)
    elif axis == 1:
        pts[:, 1] += offset
    elif axis == 0:
        pts[:, 0] += offset
    else:
        raise ValueError('Invalid axis')
    return pts

def rotate_points(points, origin, rotation):
    points_rotated = []
    for pnt in points:
        points_rotated.append(rotate(origin, pnt, rotation))
    return points_rotated

def is_outside_image(pnt, img_size, offset=0.1):
    return any(i < offset * img_size or i > (1 - offset) * img_size for i in pnt)

def get_bounding_lines(args, offset=0.1):
    min_value = args.img_size * offset
    max_value = args.img_size - min_value
    left = Line((min_value, min_value), (min_value, max_value))
    top = Line((min_value, min_value), (max_value, min_value))
    right = Line((max_value, min_value), (max_value, max_value))
    bottom = Line((max_value, max_value), (min_value, max_value))
    return [left, top, right, bottom]

def bezier_curve(left, right, bezier_pnt, img_size):
    new_pts = []
    for dt in range(img_size * 2):
        t = dt / (img_size * 2)
        p0 = left * t + (1 - t) * bezier_pnt
        p1 = bezier_pnt * t + (1 - t) * right
        p_final = p0 * t + (1 - t) * p1
        new_pts.append(tuple(p_final))
    return new_pts

def create_circle_box(center, radius):
    return [(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)]

def calc_area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

def catmull_rom_chain(P):
    """
    Calculate Catmull Rom for a chain of points and return the combined curve.
    """
    sz = len(P)
    C = [] # The curve C will contain an array of (x,y) points.
    for i in range(sz - 3):
        c = catmull_rom_spline(P[i], P[i+1], P[i+2], P[i+3])
        C.extend(c)
    return C

def catmull_rom_spline(P0, P1, P2, P3, nPoints=100):
    """
    P0, P1, P2, and P3 should be (x,y) point pairs that define the Catmull-Rom spline.
    nPoints is the number of points to include in this curve segment.
    """
    P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3]) # Convert the points to numpy so that we can do array multiplication
    alpha = 0.5 # Calculate t0 to t4
    def tj(ti, Pi, Pj):
        xi, yi = Pi
        xj, yj = Pj
        return ( ( (xj-xi)**2 + (yj-yi)**2 )**0.5 )**alpha + ti
    t0 = 0
    t1 = tj(t0, P0, P1)
    t2 = tj(t1, P1, P2)
    t3 = tj(t2, P2, P3)
    t = np.linspace(t1, t2, nPoints) # Only calculate points between P1 and P2
    t = t.reshape(len(t),1) # Reshape so that we can multiply by the points P0 to P3 and get a point for each value of t.
    A1 = (t1-t)/(t1-t0)*P0 + (t-t0)/(t1-t0)*P1
    A2 = (t2-t)/(t2-t1)*P1 + (t-t1)/(t2-t1)*P2
    A3 = (t3-t)/(t3-t2)*P2 + (t-t2)/(t3-t2)*P3
    B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
    B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3
    C  = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2
    return C

def get_angle(center, pnt):
    diff = (pnt[0] - center[0], pnt[1] - center[1])
    return math.atan2(diff[1], diff[0])

def sort_clockwise(center, pts):
    return sorted(pts, key=lambda pnt: get_angle(center, pnt))

def get_triangle_edge_lengths(pts):
    A = Point(pts[0])
    B = Point(pts[1])
    C = Point(pts[2])
    return C.distance(B), A.distance(C), A.distance(B)

def calc_triangle_area(pts):
    a, b, c = get_triangle_edge_lengths(pts)
    s = (a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area
