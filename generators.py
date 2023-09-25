# Copyright (c) 2023 IBM Research. All Rights Reserved.
#
# Code accompanying a manuscript entitled:
# "On the visual analytic intelligence of neural networks"

import utils
from geometry import Line, Point
from random import randint, uniform, choice
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import math
import numpy as np
from base_generator import BaseGenerator
from constants import RIDDLE_IDS
MIN_SLOPE_DIFF = 0.6

class ChiralGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.length = utils.get_random_of_percentage(0.3, 0.5, self.img_size)
        self.radius = utils.get_random_of_percentage(0.05, 0.1, self.img_size)
        max_pos = int(self.img_size * 0.9 - self.length // 2 - self.radius)
        min_pos = int(self.img_size * 0.1 + self.length // 2 + self.radius)
        self.x = randint(min_pos, max_pos)
        self.y = randint(min_pos, max_pos)
        self.rotation = math.radians(randint(5, 355))
        self.width = utils.get_random_of_percentage(0.01, 0.03, self.img_size)
        self.circle_offset = -self.radius if self.is_oddity else self.radius
        self.stick_length = uniform(0.3, 0.4) * self.length
        self.stick_offset = -self.stick_length if self.is_oddity else self.stick_length

    def maybe_rotate(self, pts):
        center = (self.x, self.y)
        if self.name == RIDDLE_IDS[24] or self.name == RIDDLE_IDS[25]:
            pts = utils.rotate_points(pts, center, self.rotation)
        return pts

    def draw_circle(self, d):
        circle_center = (self.x + self.circle_offset, self.y - self.length // 2)
        circle_center = self.maybe_rotate([circle_center])[0]
        ellipse_box = utils.create_circle_box(circle_center, self.radius)
        d.ellipse(ellipse_box, fill=self.color, outline=self.color)

    def draw_sticks(self, d):
        starting_pos = self.length * 0.05
        space = max(self.width * 2, self.length * 0.1)
        stick_1_pts = [(self.x, self.y - self.length // 2 + starting_pos),
                       (self.x + self.stick_offset, self.y - self.length // 2 + starting_pos)]
        stick_2_pts = [(self.x, self.y - self.length // 2 + starting_pos + space),
                       (self.x + self.stick_offset, self.y - self.length // 2 + starting_pos + space)]
        stick_1_pts = self.maybe_rotate(stick_1_pts)
        stick_2_pts = self.maybe_rotate(stick_2_pts)

        d.line(stick_1_pts, width=self.width, fill=self.color)
        d.line(stick_2_pts, width=self.width, fill=self.color)

    def generate(self, d):
        line_pts = [(self.x, self.y - self.length // 2), (self.x, self.y + self.length // 2)]
        line_pts = self.maybe_rotate(line_pts)
        d.line(line_pts, fill=self.color, width=self.width)

        if self.name == RIDDLE_IDS[22] or self.name == RIDDLE_IDS[24]:
            self.draw_circle(d)
        else:
            self.draw_sticks(d)

    def generate_oddity(self, d):
        self.generate(d)

class CircleGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.x, self.y, self.width, self.height = utils.get_random_rectangle_props(self.args)

    def generate(self, d):
        pts = [
            (self.x, self.y),
            (self.x + self.width, self.y + self.width)
        ]
        d.ellipse(pts, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        theta = np.arange(0, 2 * math.pi, 0.01)

        xpos = self.width // 2 * np.cos(theta)
        ypos = -1 * self.height // 2 * np.sin(theta)

        rotation = utils.get_random_rotation()
        pts = np.column_stack((xpos, ypos))
        pts += np.array([self.x + self.width // 2, self.y + self.height // 2])
        pts = list(map(tuple, pts))
        pts = utils.rotate_points(pts, (self.x + self.width // 2,
                                        self.y + self.height // 2), rotation)

        d.polygon(pts, fill=self.color, outline=self.color)

class CircleCenterGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.radius = uniform(0.1, 0.4)
        self.x = utils.get_random_of_percentage(self.radius, 1 - self.radius, self.img_size)
        self.y = utils.get_random_of_percentage(self.radius, 1 - self.radius, self.img_size)
        self.radius = self.radius * self.img_size
        self.width = utils.get_random_of_percentage(0.01, 0.03, self.img_size)
        self.offset = (0, 0)
        self.center_color = 'white'
        if self.is_oddity:
            while True:
                offset = (utils.get_random_multiplier() * uniform(0.2, 0.8),
                          utils.get_random_multiplier() * uniform(0.2, 0.8))
                if offset[0] * offset[0] + offset[1] * offset[1] <= 0.8 * 0.8:
                    break
            offset = (offset[0] * self.radius, offset[1] * self.radius)
            self.offset = offset

    def generate(self, d):
        pts = [ (self.x - self.radius, self.y - self.radius),
                (self.x + self.radius, self.y + self.radius) ]
        d.ellipse(pts, fill=self.color, outline=self.color)
        center_box = [  (self.x - self.width + self.offset[0], self.y - self.width + self.offset[1]),
                        (self.x + self.width + self.offset[0], self.y + self.width + self.offset[1]),]
        d.ellipse(center_box, fill=self.center_color, outline=self.center_color)

    def generate_oddity(self, d):
        self.generate(d)

class ClosureGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.center = utils.random_point(self.img_size, offset=0.2) 

        while True:
            top_left = (uniform(0.1 * self.img_size, self.center[0]), uniform(0.1 * self.img_size, self.center[1]))
            top_right = (uniform(self.center[0], 0.9 * self.img_size), uniform(0.1 * self.img_size, self.center[1]))
            bottom_right = (uniform(self.center[0], 0.9 * self.img_size), uniform(self.center[1], 0.9 * self.img_size))
            bottom_left = (uniform(0.1 * self.img_size, self.center[0]), uniform(self.center[1], 0.9 * self.img_size))

            pts = [top_left, top_right, bottom_right, bottom_left]
            num_pts = randint(4, 8) 
            pts.extend(utils.random_points(num_pts, self.img_size))
            self.pts = utils.sort_clockwise(self.center, pts)
            if utils.calc_area(self.pts) > 0.25 * self.img_size * 0.25 * self.img_size:
                break
        
        if not self.is_oddity:
            self.pts.extend(self.pts[:2])
        self.pts = utils.catmull_rom_chain(self.pts)
        self.pts = list(map(tuple, self.pts))

    def generate(self, d):
        d.polygon(self.pts, outline=self.color)

    def generate_oddity(self, d):
        d.line(self.pts, fill=self.color)

class ColorGenerator(BaseGenerator):
    # We generate a random rectangle and then fill it for oddities
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        x, y, width, height = utils.get_random_rectangle_props(self.args)
        self.state = {'pts': [x, y, x + width, y + height]}

    def generate(self, d):
        d.rectangle(self.state['pts'], outline=self.color)

    def generate_oddity(self, d):
        d.rectangle(self.state['pts'], fill=self.color, outline=self.color)

class ConnectednessGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        if self.is_oddity:
            return self.set_oddity_state()
        while True:
            num_pts = randint(8, 12) 
            pts = utils.random_points(num_pts, self.img_size)
            center = pts.mean(axis=0)
            self.pts = utils.sort_clockwise(center, pts)
            if utils.calc_area(self.pts) > 0.25 * self.img_size * 0.25 * self.img_size:
                break
        self.pts.extend(self.pts[:2])
        self.pts = utils.catmull_rom_chain(self.pts)
        self.pts = list(map(tuple, self.pts))

    def set_oddity_state(self):
        props = utils.get_random_line_props(self.args)
        mid = self.img_size // 2
        line_pts = [ (mid - props['length'] // 2, mid), (mid + props['length'] // 2, mid) ]
        line_pts = utils.rotate_points(line_pts, (mid, mid), props['rotation'])

        def get_sign(pnt):
            x1, y1 = line_pts[0]
            x2, y2 = line_pts[1]
            x, y = pnt
            diff = self.img_size * 0.05
            d_upper_bound = (x - x1) * ((y2 + diff) - (y1 + diff)) - (y - (y1 + diff)) * (x2 - x1)
            d_lower_bound = (x - x1) * ((y2 - diff) - (y1 - diff)) - (y - (y1 - diff)) * (x2 - x1)
            if d_lower_bound < 0 and d_upper_bound < 0:
                return -1
            elif d_lower_bound > 0 and d_lower_bound > 0:
                return 1
            return 0

        num_left = randint(8, 12)
        num_right = randint(8, 12)
        left_pts = []
        right_pts = []
        while len(right_pts) < num_right or len(left_pts) < num_left:
            pnt = utils.random_point(self.img_size)
            sign = get_sign(pnt)
            if not sign:
                continue
            elif sign == 1 and len(right_pts) < num_right and pnt not in right_pts:
                right_pts.append(pnt)
            elif sign == -1 and len(left_pts) < num_left and pnt not in left_pts:
                left_pts.append(pnt)
        left_center = np.array(left_pts).mean(axis=0)
        right_center = np.array(right_pts).mean(axis=0)
        self.right_pts = utils.sort_clockwise(right_center, right_pts)
        self.left_pts = utils.sort_clockwise(left_center, left_pts)
        self.left_pts.extend(self.left_pts[:2])
        self.right_pts.extend(self.right_pts[:2])
        self.left_pts = utils.catmull_rom_chain(self.left_pts)
        self.left_pts = list(map(tuple, self.left_pts))
        self.right_pts = utils.catmull_rom_chain(self.right_pts)
        self.right_pts = list(map(tuple, self.right_pts))

    def generate(self, d):
        d.polygon(self.pts, outline=self.color, fill=self.color)

    def generate_oddity(self, d):
        d.polygon(self.right_pts, outline=self.color, fill=self.color)
        d.polygon(self.left_pts, outline=self.color, fill=self.color)

class ConvexGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        while True:
            self.pts = utils.random_points(10, self.img_size)
            self.convex_hull = ConvexHull(np.array(self.pts))
            if self.convex_hull.volume > (self.img_size * 0.1) ** 2:
                break
        self.pts = np.array(self.pts[self.convex_hull.vertices])
        mean_pnt = np.mean(self.pts, axis=0)
        swap_id = randint(0, len(self.pts) - 1)
        if self.is_oddity:
            self.pts = np.concatenate((self.pts[:swap_id], [mean_pnt], self.pts[swap_id:]), axis=0)
        self.pts = np.concatenate((self.pts, self.pts[:2]), axis=0)
        self.pts = utils.catmull_rom_chain(self.pts)
        self.pts = list(map(tuple, self.pts))

    def generate(self, d):
        d.polygon(self.pts, outline=self.color, fill=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class DistanceGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_global_state(self):
        self.distance = uniform(0.05, 0.2)
        self.width = utils.get_random_of_percentage(0.01, 0.03, self.img_size)

    def set_state(self):
        self.props = utils.get_random_line_props(self.args)
        if self.is_oddity:
            while True:
                distance = uniform(0.05, 0.25)
                if abs(self.distance - distance) > 0.07:
                    self.distance = distance
                    break

    def generate(self, d):
        pts = [ (self.props['x'] - self.props['length'] // 2, self.props['y']),
                (self.props['x'] + self.props['length'] // 2, self.props['y']),
                (self.props['x'], self.props['y'] + self.distance * self.img_size) ]
        pts = utils.rotate_points(pts, (self.props['x'], self.props['y']), self.props['rotation'])
        d.line(pts[:2], fill=self.color, width=self.width)
        ellipse_box = [(pts[2][0] - self.width, pts[2][1] - self.width),
                       (pts[2][0] + self.width, pts[2][1] + self.width)]
        d.ellipse(ellipse_box, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class EquilateralGenerator(BaseGenerator):
    # Generate a random edge length. The oddity needs to have one
    # edge with a different edge length. Pythogoras rule used to find height of triangle.
    #         A
    #         /\
    #        /  \
    #       /    \
    #      /   M  \
    #   B /________\ C
    # M_y is located 2/3 * height from A and 1/3 * height from B and C.
    # M_x is located 0 from A and 1/2 * |BC| from B and C
    # We then rotate it around the midpoint before calculating the
    # maximum offset it can have before crashing into the 0.1 box
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_random_edge_length(self):
        return utils.get_random_of_percentage(0.25, 0.5, self.img_size)

    def set_state(self):
        self.rotation = utils.get_random_rotation()
        self.edge_length = self.get_random_edge_length()
        self.side_edge_length = self.edge_length
        if self.is_oddity:
            self.side_edge_length = self.get_random_edge_length()
            while abs(self.side_edge_length - self.edge_length) < 0.1 * self.img_size:
                self.side_edge_length = self.get_random_edge_length()

    def generate(self, d):
        height = math.sqrt(4 * math.pow(self.side_edge_length, 2) -
                           math.pow(self.edge_length, 2)) / 2
        center_to_top = (0, 2 * height / 3)
        center_to_side = (self.edge_length / 2, height / 3)
        mid = self.img_size / 2
        pts = [ (mid, mid - center_to_top[1]),
                (mid + center_to_side[0], mid + center_to_side[1]),
                (mid - center_to_side[0], mid + center_to_side[1]), ]
        pts = utils.rotate_points(pts, (mid, mid), self.rotation)
        min_x, min_y, max_x, max_y = utils.get_minmax_offsets(pts, self.img_size)
        x_offset = uniform(-min_x, max_x)
        y_offset = uniform(-min_y, max_y)
        pts = utils.translate_points(pts, (x_offset, y_offset)).flatten().tolist()
        d.polygon(pts, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class FixedGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_global_state(self):
        self.factor = uniform(1.3, 3)

    def get_adjustment_translation(self, pts):
        bbox = utils.get_bbox(pts)
        t_x = 0
        t_y = 0
        noise_x = uniform(0, self.img_size * 0.1)
        noise_y = uniform(0, self.img_size * 0.1)
        if bbox[0] < self.img_size * 0.1 + noise_x:
            t_x = self.img_size * 0.1 + noise_x - bbox[0]
        if bbox[1] < self.img_size * 0.1 + noise_y:
            t_y = self.img_size * 0.1 + noise_y - bbox[1]
        return t_x, t_y

    def set_state(self):
        if self.is_oddity and self.name == RIDDLE_IDS[35]:
            factor = uniform(1.3, 5)
            while abs(self.factor - factor) < 1.5:
                factor = uniform(1.3, 5)
            self.factor = factor

        while True:
            pts = utils.random_points(3, self.img_size)
            area = utils.calc_triangle_area(pts)
            if not isinstance(area, complex) and area > (0.4 * self.img_size) ** 2 / 2:
                break
        min_x_i = np.argmin(pts[:, 0])
        large_pts = [pts[min_x_i]]
        for i in range(3):
            if i == min_x_i:
                continue
            diff_vec = pts[i] - pts[min_x_i]
            large_pnt = pts[min_x_i] + self.factor * diff_vec
            large_pts.append(large_pnt)

        min_diff = 0.05 * self.img_size
        if self.is_oddity and self.name == RIDDLE_IDS[34]:
            center = ((pts[0][0] + pts[1][0] + pts[2][0]) / 3, (pts[0][1] + pts[1][1] + pts[2][1]) / 3)
            pts = np.array(utils.rotate_points(pts, center, math.radians(uniform(15, 345))))
        bb_w, bb_h = utils.get_bbox_lengths(pts)
        bb_large_w, bb_large_h = utils.get_bbox_lengths(large_pts)

        min_right_t = bb_w + min_diff
        min_left_t = bb_large_w + min_diff
        min_y_i = np.argmin(pts[:, 1])
        max_y_i = np.argmax(pts[:, 1])
        if min_y_i == min_x_i:
            min_up_t = bb_h
            min_down_t = bb_large_h
        elif max_y_i == min_x_i:
            min_up_t = bb_large_h
            min_down_t = bb_h
        else:
            min_up_t = bb_large_h
            min_down_t = bb_large_h
        min_up_t += min_diff
        min_down_t += min_diff
        while True: # make sure translation is enough to avoid intersection
            angle = math.radians(uniform(0, 360))
            distance = uniform(min(bb_h, bb_w) + min_diff, 1.5 * max(bb_large_h, bb_large_w))
            d_x = distance * math.cos(angle)
            d_y = distance * math.sin(angle)
            if d_x > 0 and d_y > 0:
                if d_x > min_right_t or d_y > min_up_t:
                    break
            if d_x > 0 and d_y < 0:
                if d_x > min_right_t or d_y < -min_down_t:
                    break
            if d_x < 0 and d_y > 0:
                if d_x < -min_left_t or d_y > min_up_t:
                    break
            if d_x < 0 and d_y < 0:
                if d_x < -min_left_t or d_y < -min_down_t:
                    break
        large_pts = utils.translate_points(large_pts, (d_x, d_y))
        offset = self.get_adjustment_translation(large_pts)
        large_pts = utils.translate_points(large_pts, offset)
        pts = utils.translate_points(pts, offset)
        max_value = max(max(utils.get_bbox(pts)), max(utils.get_bbox(large_pts)))
        factor = max_value / (self.img_size * 0.9)
        pts = pts / factor
        large_pts = large_pts / factor
        self.large_pts = large_pts.flatten().tolist()
        self.pts = pts.flatten().tolist()

    def generate(self, d):
        d.polygon(self.pts, fill=self.color, outline=self.color)
        d.polygon(self.large_pts, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class HolesGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        while True:
            self.set_pts()
            if utils.calc_area(self.inside_pts) > (0.1 * self.img_size) ** 2 or not self.is_oddity:
                break

    def set_pts(self):
        bb_width = utils.get_random_of_percentage(0.2, 0.4, self.img_size)
        bb_height = utils.get_random_of_percentage(0.2, 0.4, self.img_size)
        bb_x = utils.get_random_of_percentage(0.3, 0.7, self.img_size) - bb_width / 2
        bb_y = utils.get_random_of_percentage(0.3, 0.7, self.img_size) - bb_height / 2
        num_points = randint(10, 20)
        while True:
            inside_pts = []
            while len(inside_pts) < num_points:
                x = uniform(bb_x, bb_x + bb_width)
                y = uniform(bb_y, bb_y + bb_height)
                inside_pts.append((x, y))
            if utils.calc_area(inside_pts) >= (0.2 * self.img_size) ** 2:
                break
        diff = 0.05 * self.img_size
        # in order to make sure the polygon covers the inside polygon,
        # we need there to be a point in each corner box of the inner bounding box
        top_left = (uniform(0.1 * self.img_size, bb_x - diff), uniform(0.1 * self.img_size, bb_y - diff))
        top_right = (uniform(bb_x + bb_width + diff, 0.9 * self.img_size), uniform(0.1 * self.img_size, bb_y - diff))
        bottom_right = (uniform(bb_x + bb_width + diff, 0.9 * self.img_size), uniform(bb_y + bb_height + diff, 0.9 * self.img_size))
        bottom_left = (uniform(0.1 * self.img_size, bb_x - diff), uniform(bb_y + bb_height + diff, 0.9 * self.img_size))
        pts = [top_left, top_right, bottom_right, bottom_left]
        selections = [  [0.1 * self.img_size, bb_x - diff, 0.1 * self.img_size, 0.9 * self.img_size],
                        [bb_x, bb_x + bb_width, 0.1 * self.img_size, bb_y - diff],
                        [bb_x, bb_x + bb_width, bb_y + bb_height + diff, 0.9 * self.img_size],
                        [bb_x + bb_width + diff, 0.9 * self.img_size, 0.1 * self.img_size, 0.9 * self.img_size] ]
        num_points = randint(10, 20)
        while len(pts) < num_points:
            selection = choice(selections)
            pnt = (uniform(selection[0], selection[1]), uniform(selection[2], selection[3]))
            pts.append(pnt)
        self.pts = utils.sort_clockwise((bb_x + bb_width / 2, bb_y + bb_height / 2), pts)
        self.inside_pts = utils.sort_clockwise((bb_x + bb_width / 2, bb_y + bb_height / 2), inside_pts)
        self.pts.extend(self.pts[:2])
        self.pts = utils.catmull_rom_chain(self.pts)
        self.pts = list(map(tuple, self.pts))
        self.inside_pts.extend(self.inside_pts[:2])
        self.inside_pts = utils.catmull_rom_chain(self.inside_pts)
        self.inside_pts = list(map(tuple, self.inside_pts))

    def generate(self, d):
        d.polygon(self.pts, outline=self.color, fill=self.color)

    def generate_oddity(self, d):
        self.generate(d)
        d.polygon(self.inside_pts, fill='white', outline='white') # todo: replace with background color

class InsideGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_oddity_pts(self):
        quarter = randint(0, 3)
        offset = 0.1
        box = 0.25
        q_fracs = [     [1 - box, 1 - offset, offset, box],
                        [offset, box, offset, box],
                        [offset, box, 1 - box, 1 - offset],
                        [1 - box, 1 - offset, 1 - box, 1 - offset]  ]
        x1 = utils.get_random_of_percentage(q_fracs[quarter][0], q_fracs[quarter][1], self.img_size)
        y1 = utils.get_random_of_percentage(q_fracs[quarter][2], q_fracs[quarter][3], self.img_size)
        multiplier = 1 if quarter % 2 else -1
        x2 = x1 + 1
        y2 = y1 + multiplier
        def get_sign(pnt):
            x, y = pnt
            d = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            return d >= 0
        corner = [[1, 0], [0, 0], [0, 1], [1, 1]][quarter]
        corner_sign = get_sign((corner[0] * self.img_size, corner[1] * self.img_size))
        num_pts = randint(8, 12)
        pts = []
        while len(pts) < num_pts:
            pnt = utils.random_point(self.img_size)
            pnt_sign = get_sign(pnt)
            if pnt_sign != corner_sign:
                pts.append(pnt)
        outside_pnt = (x1, y1)
        mean_pnt = np.array(pts).mean(axis=0)
        pts = utils.sort_clockwise(mean_pnt, pts)
        return outside_pnt, pts

    def set_state(self):
        self.inside_pnt = utils.random_point(self.img_size, offset=0.2)
        if self.is_oddity:
            self.inside_pnt, self.pts = self.get_oddity_pts()
        while True and not self.is_oddity:
            top_left = (uniform(0.1 * self.img_size, self.inside_pnt[0]), uniform(0.1 * self.img_size, self.inside_pnt[1]))
            top_right = (uniform(self.inside_pnt[0], 0.9 * self.img_size), uniform(0.1 * self.img_size, self.inside_pnt[1]))
            bottom_right = (uniform(self.inside_pnt[0], 0.9 * self.img_size), uniform(self.inside_pnt[1], 0.9 * self.img_size))
            bottom_left = (uniform(0.1 * self.img_size, self.inside_pnt[0]), uniform(self.inside_pnt[1], 0.9 * self.img_size))
            pts = [top_left, top_right, bottom_right, bottom_left]
            num_pts = randint(4, 8) 
            pts.extend(utils.random_points(num_pts, self.img_size))
            self.pts = utils.sort_clockwise(self.inside_pnt, pts)
            if utils.calc_area(self.pts) > 0.25 * self.img_size * 0.25 * self.img_size:
                break
        self.pts.extend(self.pts[:2])
        self.pts = utils.catmull_rom_chain(self.pts)
        self.pts = list(map(tuple, self.pts))

    def generate(self, d):
        d.polygon(self.pts, outline=self.color)
        point_box = utils.create_circle_box(self.inside_pnt, 2)
        d.ellipse(point_box, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class LineGenerator(BaseGenerator):
    # Straight line: Line props generated, then an offset point is set for the bezier curve
    # Curve: Same as straight but opposite
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.state = {'props': utils.get_random_line_props(self.args)}

    def generate_curved_line(self, d):
        props = self.state['props']
        new_pts = []
        dx = utils.get_random_multiplier() * utils.get_random_of_percentage(0, 0.1, self.img_size)
        dy = utils.get_random_multiplier() * utils.get_random_of_percentage(0.1, 0.25, self.img_size)
        bezier_point = np.array((props['x'] + dx, props['y'] + dy))
        left = np.array(props['pts'][0])
        right = np.array(props['pts'][1])
        for dt in range(self.img_size * 2):
            t = dt / (self.img_size * 2)
            p0 = left * t + (1 - t) * bezier_point
            p1 = bezier_point * t + (1 - t) * right
            p_final = p0 * t + (1 - t) * p1
            new_pts.append(tuple(p_final))
        new_pts = utils.rotate_points(new_pts, (props['x'], props['y']), props['rotation'])
        d.line(new_pts, fill=self.color, width=props['width'])

    def generate_straight_line(self, d):
        props = self.state['props']
        pts = utils.rotate_points(props['pts'], (props['x'], props['y']), props['rotation'])
        d.line(pts, fill=self.color, width=props['width'])

    def generate(self, d):
        if self.name == RIDDLE_IDS[2]:
            self.generate_straight_line(d)
        elif self.name == RIDDLE_IDS[3]:
            self.generate_curved_line(d)

    def generate_oddity(self, d):
        if self.name == RIDDLE_IDS[3]:
            self.generate_straight_line(d)
        elif self.name == RIDDLE_IDS[2]:
            self.generate_curved_line(d)

class OrientationGenerator(BaseGenerator):
    # We generate a random rectangle and then rotate them for oddities
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.x, self.y, self.width, self.height = utils.get_random_rectangle_props(self.args)

    def generate(self, d):
        d.rectangle([self.x, self.y, self.x + self.width, self.y + self.height], fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        rotation_multiplier = utils.get_random_multiplier()
        rotation = math.radians(rotation_multiplier * randint(15, 45))
        origin_of_rotation = (self.x + self.width // 2, self.y + self.height // 2)
        points = [(self.x, self.y),
                  (self.x + self.width, self.y),
                  (self.x + self.width, self.y + self.height),
                  (self.x, self.y + self.height)]
        points_rotated = utils.rotate_points(points, origin_of_rotation, rotation)
        d.polygon(points_rotated, fill=self.color, outline=self.color)

class ParallelGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_parallel_lines(self, d):
        props = utils.get_random_line_props(self.args)
        space = utils.get_random_of_percentage(0.05, 0.15, self.img_size)
        upper_pts = utils.translate_points(props['pts'], -space // 2, axis=1)
        lower_pts = utils.translate_points(props['pts'], space // 2, axis=1)
        d.line(utils.rotate_points(upper_pts, (props['x'], props['y']), props['rotation']), fill=self.color, width=props['width'])
        d.line(utils.rotate_points(lower_pts, (props['x'], props['y']), props['rotation']), fill=self.color, width=props['width'])

    def generate_secant_lines(self, d):
        props = utils.get_random_line_props(self.args)
        space = utils.get_random_of_percentage(0.10, 0.25, self.img_size)
        upper_pts = utils.translate_points(props['pts'], -space // 2, axis=1)
        lower_pts = utils.translate_points(props['pts'], space // 2, axis=1)
        rotation_degrees = int(math.degrees(props['rotation']))
        new_rotation = randint(rotation_degrees - 30, rotation_degrees + 30) % 180
        while abs(new_rotation - rotation_degrees) < 10:
            new_rotation = randint(rotation_degrees - 30, rotation_degrees + 30) % 180
        d.line(utils.rotate_points(upper_pts, (props['x'], props['y']), props['rotation']), fill=self.color, width=props['width'])
        d.line(utils.rotate_points(lower_pts, (props['x'], props['y']), math.radians(new_rotation)), fill=self.color, width=props['width'])

    def generate(self, d):
        if self.name == RIDDLE_IDS[6]:
            self.generate_parallel_lines(d)
        elif self.name == RIDDLE_IDS[7]:
            self.generate_secant_lines(d)

    def generate_oddity(self, d):
        if self.name == RIDDLE_IDS[6]:
            self.generate_secant_lines(d)
        elif self.name == RIDDLE_IDS[7]:
            self.generate_parallel_lines(d)

class ParallelogramGenerator(BaseGenerator):
    # Generate random height and width. Then calculate the minimum angle such that the polygon will not reach out of the 0.1 box
    # We rotate the polygon and then calculate how much it can be moved without going outside the box
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_offsets(self):
        left_angle = self.get_random_angle()
        if self.is_oddity:
            right_angle = self.get_random_angle()
            self.left_x_offset = self.calc_offset(left_angle)
            self.right_x_offset = self.calc_offset(right_angle)
            # second constraint is so we don't get a side with negative size when using offsets
            while abs(left_angle - right_angle) < 10 or self.width - self.right_x_offset + self.left_x_offset < 0.1 * self.img_size:
                left_angle = self.get_random_angle()
                right_angle = self.get_random_angle()
                self.left_x_offset = self.calc_offset(left_angle)
                self.right_x_offset = self.calc_offset(right_angle)
        else:
            right_angle = left_angle
            self.left_x_offset = self.calc_offset(left_angle)
            self.right_x_offset = self.calc_offset(right_angle)

    def set_state(self):
        self.width = utils.get_random_of_percentage(0.2, 0.35, self.img_size)
        self.height = utils.get_random_of_percentage(0.2, 0.35, self.img_size)
        while True:
            self.set_offsets()
            if self.height + max(0, self.left_x_offset) + max(0, self.right_x_offset) <= 0.8 * self.img_size:
                break
        self.rotation = utils.get_random_rotation()

    def get_random_angle(self):  # tan(a) = h / offset
        max_offset = 0.4 * self.img_size - self.width / 2
        min_angle = math.atan(self.height / max_offset)
        min_angle = math.ceil(math.degrees(min_angle))
        return randint(min_angle, 180 - min_angle)

    def calc_offset(self, angle):
        return self.height / math.tan(math.radians(angle))

    def generate(self, d):
        mid = self.img_size // 2
        pts = [ (mid - self.width // 2, mid - self.height // 2),
                (mid + self.width // 2, mid - self.height // 2),
                (mid + self.width // 2 - self.right_x_offset, mid + self.height // 2),
                (mid - self.width // 2 - self.left_x_offset, mid + self.height // 2)]
        pts = utils.rotate_points(pts, (mid, mid), self.rotation)
        min_x = np.amin(np.array(pts)[:, 0])
        min_y = np.amin(np.array(pts)[:, 1])
        max_x = np.amax(np.array(pts)[:, 0])
        max_y = np.amax(np.array(pts)[:, 1])
        max_x_offset = self.img_size * 0.9 - max_x
        min_x_offset = min_x - self.img_size * 0.1
        max_y_offset = self.img_size * 0.9 - max_y
        min_y_offset = min_y - self.img_size * 0.1
        x = randint(-int(min_x_offset), int(max_x_offset))
        y = randint(-int(min_y_offset), int(max_y_offset))
        pts = np.array(pts) + np.array([x, y])
        d.polygon(pts.flatten().tolist(), fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class PointOnLineGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        outside = self.name == RIDDLE_IDS[5]
        frame_offset = 0.1 if outside else 0
        props = utils.get_random_line_props(self.args, offset=frame_offset)
        pts = utils.rotate_points(props['pts'], (props['x'], props['y']), props['rotation'])
        y_offset = 0
        if self.is_oddity:
            y_offset = utils.get_random_multiplier() * utils.get_random_of_percentage(0.1, 0.15, self.img_size)
        if outside:
            x_offset = utils.get_random_multiplier() *\
                (props['length'] // 2 + utils.get_random_of_percentage(0.1, 0.15, self.img_size))
        else:
            x_offset = randint(-props['length'] // 2 + int(props['length'] * 0.2), props['length'] // 2 - int(props['length'] * 0.2))
        pnt = (props['x'] + x_offset, props['y'] + y_offset)
        pnt = utils.rotate((props['x'], props['y']), pnt, props['rotation'])
        point_radius = props['width']
        ellipse_box = utils.create_circle_box(pnt, point_radius)
        self.props = props
        self.pts = pts
        self.ellipse_box = ellipse_box

    def generate(self, d):
        d.line(self.pts, fill=self.color, width=self.props['width'])
        d.ellipse(self.ellipse_box, outline=self.color, fill=self.color)

    def generate_oddity(self, d):
        d.line(self.pts, fill=self.color, width=self.props['width'])
        d.ellipse(self.ellipse_box, outline=self.color, fill=self.color)

class PointSymmetryGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        mid = self.img_size / 2
        while True:
            pts = []
            while len(pts) < 3:
                pnt = (uniform(0.1 * self.img_size, 0.9 * self.img_size), uniform(0.1 * self.img_size, mid - 0.05 * self.img_size))
                pts.append(pnt)
            if utils.calc_triangle_area(pts) >= (self.img_size * 0.25) ** 2 / 2:
                break
        if self.is_oddity:
            pts_2 = utils.translate_points(pts, mid, axis=1)
        else:
            pts_2 = np.array(pts)
            pts_2[:] = np.ones((3, 2)) * self.img_size - pts_2[:]
        rotation = math.radians(uniform(0, 360))
        self.pts = utils.rotate_points(pts, (mid, mid), rotation)
        self.pts_2 = utils.rotate_points(pts_2, (mid, mid), rotation)
        self.point_box = utils.create_circle_box((mid, mid), uniform(0.01 * self.img_size, 0.04 * self.img_size))
        min_value = np.amin(np.array([self.pts, self.pts_2]))
        if min_value < 0.1 * self.img_size:
            utils.translate_points(self.pts, self.img_size * 0.1 - min_value)
            utils.translate_points(self.pts_2, self.img_size * 0.1 - min_value)
        max_value = np.amax(np.array([self.pts, self.pts_2]))
        if max_value > 0.9 * self.img_size:
            factor = max_value / (0.9 * self.img_size)
            self.pts = np.array(self.pts) / factor
            self.pts_2 = np.array(self.pts_2) / factor
        self.pts = np.array(self.pts).flatten().tolist()
        self.pts_2 = np.array(self.pts_2).flatten().tolist()

    def generate(self, d):
        d.ellipse(self.point_box, fill=self.color, outline=self.color)
        d.polygon(self.pts, fill=self.color, outline=self.color)
        d.polygon(self.pts_2, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class PointDistanceGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.props = utils.get_random_line_props(self.args)
        self.radius = utils.get_random_of_percentage(0.01, 0.03, self.img_size)
        self.positions = self.get_positions()

    def get_positions(self):
        if self.name == RIDDLE_IDS[28]:
            return self.get_increasing_positions()
        elif self.name == RIDDLE_IDS[27]:
            return self.get_equidistance_positions()
        else:
            raise ValueError('Incorrect RIDDLE name')

    def get_increasing_positions(self):   # 1 - x2 >= x2 - x1 >= x1
        x1 = uniform(0.1, 0.25)
        x2 = uniform(2 * x1 + 0.05, (0.95 + x1) / 2)
        if self.is_oddity:
            x1 = uniform(0.1, 0.8)
            bounds = self.get_bounds(x1)
            x2 = uniform(*choice(bounds))
        return [0, x1, x2, 1]

    def get_bounds(self, x1):
        first = 2 * x1
        second = 0.5 + 0.5 * x1
        lower = min(first, second)
        upper = min(first, second)
        bounds = [[x1 + 0.1, lower]]
        if first < 0.9:
            bounds.append([upper, 0.9])
        legal_bounds = []
        for bound in bounds:
            if bound[1] > bound[0]:
                legal_bounds.append(bound)
        assert len(legal_bounds) > 0
        return legal_bounds

    def get_equidistance_positions(self):
        positions = [0, 1 / 3, 2 / 3, 1.0]
        if self.is_oddity:
            offset = utils.get_random_multiplier() * uniform(0.1, 0.12)
            if randint(0, 1):
                positions[2] += offset
                positions[1] = self.positions[2] / 2
            else:
                positions[1] += offset
                positions[2] = self.positions[1] * 2

        return positions

    def generate(self, d):
        pts = []
        for pos in self.positions:
            pts.append((self.props['pts'][0][0] + pos * self.props['length'], self.props['y']))
        pts = utils.rotate_points(pts, (self.props['x'], self.props['y']), self.props['rotation'])
        for pnt in pts:
            ellipse_box = utils.create_circle_box(pnt, self.radius)
            d.ellipse(ellipse_box, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class ProportionsGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def set_global_state(self):
        if self.name == RIDDLE_IDS[31]:
            self.proportion = 0.5
        else:
            self.proportion = uniform(0.1, 0.9)

    def get_oddity_proportion(self):
        if self.name == RIDDLE_IDS[31]:
            new_pos = uniform(0.1, 0.35)
            new_pos = abs(randint(0, 1) - new_pos)
            return new_pos
        else:
            while True:
                new_pos = uniform(0.1, 0.9)
                if abs(new_pos - self.proportion) > 0.2:
                    return new_pos

    def set_state(self):
        self.props = utils.get_random_line_props(self.args)
        self.width = utils.get_random_of_percentage(0.01, 0.02, self.img_size)
        if self.is_oddity:
            self.proportion = self.get_oddity_proportion()
        
        self.line_pts = self.props['pts']
        self.pnt = (self.props['pts'][0][0] + self.props['length'] * self.proportion, self.props['y'])

        if self.name == RIDDLE_IDS[31]:
            self.line_pts = utils.rotate_points(self.props['pts'], self.props['center'], self.props['rotation'])
            self.pnt = utils.rotate_points([self.pnt], self.props['center'], self.props['rotation'])[0]

    def generate(self, d):
        d.line(self.line_pts, width=self.width, fill=self.color)
        point_box = utils.create_circle_box(self.pnt, self.width)
        d.ellipse(point_box, outline=self.color, fill=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class QuadrilateralGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        area_large_enough = False
        while not area_large_enough:
            pts = utils.random_points(50, self.img_size, offset=0.35)
            x_offset = pts[:, 0]
            y_offset = pts[:, 1]
            left_upper = np.argmin(x_offset + y_offset)
            y_offset = -pts[:, 1] + self.img_size
            left_lower = np.argmin(x_offset + y_offset)
            x_offset = -pts[:, 0] + self.img_size
            right_lower = np.argmin(x_offset + y_offset)
            y_offset = pts[:, 1]
            right_upper = np.argmin(x_offset + y_offset)
            indices = [left_upper, right_upper, right_lower, left_lower]
            min_length = self.img_size * 0.05
            area_large_enough = pts[right_upper, 0] - pts[left_upper, 0] > min_length and \
                pts[right_lower, 1] - pts[right_upper, 1] > min_length and \
                pts[right_lower, 0] - pts[left_lower, 0] > min_length and \
                pts[left_lower, 1] - pts[left_upper, 1] > min_length
        offset = np.array([uniform(-0.25, 0.25), uniform(-0.25, 0.25)])
        offset *= self.img_size
        pts = pts + offset
        if self.is_oddity:
            index_to_remove = randint(0, 3)
            indices.remove(indices[index_to_remove])
        pts = utils.rotate_points(pts, (self.img_size * 0.5 + offset[0], self.img_size * 0.5 + offset[1]), utils.get_random_rotation())
        pts = np.array(pts)
        polygon_pts = pts[indices].flatten().tolist()
        self.pts = polygon_pts

    def generate(self, d):
        d.polygon(self.pts, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class QuadrilateralCenterGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_allowed(self, angles, angle):
        diff = 20
        for point in angles:
            if point + diff > 360 or point - diff < 0:
                if not (angle >= (point + diff) % 360 and angle <= (point + 360 - diff) % 360):
                    return False
            if angle > point - diff and angle < point + diff:
                return False
        return True

    def get_pts(self):
        radius = uniform(0.1, 0.4)
        angles = []
        while len(angles) < 4:
            angle = randint(0, 360)
            if self.is_allowed(angles, angle):
                angles.append(angle)
        angles = sorted(angles)
        x = utils.get_random_of_percentage(radius + 0.1, 0.9 - radius, self.img_size)
        y = utils.get_random_of_percentage(radius + 0.1, 0.9 - radius, self.img_size)
        radius *= self.img_size
        pts = []
        for angle in angles:
            pts.append((x + radius * math.cos(math.radians(angle)), y + radius * math.sin(math.radians(angle))))
        return pts

    def get_center(self):
        line1 = Line(self.pts[0], self.pts[2])
        line2 = Line(self.pts[1], self.pts[3])
        return line1.intersection(line2)

    def get_diff_vec(self, pnt):
        travel = uniform(0.3, 0.7)
        diff_vec = (pnt[0] - self.center[0], pnt[1] - self.center[1])
        diff = (diff_vec[0] * travel, diff_vec[1] * travel)
        return diff

    def translate_center(self):
        distances = []
        center = self.center
        for i, pnt in enumerate(self.pts):
            distances.append((center.distance(pnt), i))
        distances = sorted(distances)
        diff1 = self.get_diff_vec(self.pts[distances[2][1]])
        diff2 = self.get_diff_vec(self.pts[distances[3][1]])
        mid = ((diff1[0] + diff2[0]) / 2, (diff1[1] + diff2[1]) / 2)
        self.center = (self.center[0] + mid[0], self.center[1] + mid[1])

    def set_state(self):
        self.center_color = 'white'
        while True:
            pts = self.get_pts()
            if utils.calc_area(pts) > 0.25 * self.img_size * 0.25 * self.img_size:
                self.pts = pts
                break
        self.center = self.get_center()
        self.width = utils.get_random_of_percentage(0.01, 0.02, self.img_size)
        if self.is_oddity:
            self.translate_center()

    def generate(self, d):
        d.polygon(self.pts, fill=self.color, outline=self.color)
        center_box = utils.create_circle_box(self.center, self.width)
        d.ellipse(center_box, fill=self.center_color, outline=self.center_color)

    def generate_oddity(self, d):
        self.generate(d)

class RectangleGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.rotation = utils.get_random_rotation()
        self.width = utils.get_random_of_percentage(0.2, 0.35, self.img_size)
        self.height = utils.get_random_of_percentage(0.2, 0.35, self.img_size)

    def get_random_angle(self):
        min_angle = 44  # so the diameter will never be >= 0.8
        max_angle = 75  # so it won't look like a rectangle
        return (180 + utils.get_random_multiplier() * randint(min_angle, max_angle)) % 180

    def calc_offset(self, angle):
        return self.height / math.tan(math.radians(angle))

    def generate(self, d):
        x, y, _, _ = utils.get_random_rectangle_props(self.args)
        origin_of_rotation = (x + self.width // 2, y + self.height // 2)
        points = [(x, y), (x + self.width, y), (x + self.width, y + self.height), (x, y + self.height)]
        points_rotated = utils.rotate_points(points, origin_of_rotation, self.rotation)
        d.polygon(points_rotated, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        angle = self.get_random_angle()
        self.x_offset = self.calc_offset(angle)
        mid = self.img_size // 2
        pts = [ (mid - self.width // 2, mid - self.height // 2),
                (mid + self.width // 2, mid - self.height // 2),
                (mid + self.width // 2 - self.x_offset, mid + self.height // 2),
                (mid - self.width // 2 - self.x_offset, mid + self.height // 2) ]
        pts = utils.rotate_points(pts, (mid, mid), self.rotation)
        min_x_offset, min_y_offset, max_x_offset, max_y_offset = utils.get_minmax_offsets(pts, self.img_size)
        x = randint(-int(min_x_offset), int(max_x_offset))
        y = randint(-int(min_y_offset), int(max_y_offset))
        pts = np.array(pts) + np.array([x, y])
        d.polygon(pts.flatten().tolist(), fill=self.color, outline=self.color)

class RightAngleGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.props = utils.get_random_line_props(self.args)
        self.props_orthogonal = utils.get_random_line_props(self.args)
        self.props_orthogonal.update(self.props)
        self.props_orthogonal['pts'] = self.props['pts'][:]  # clone
        if self.is_oddity:
            self.props_orthogonal['rotation'] += utils.get_random_multiplier() * \
                math.radians(randint(25, 65))
        else:
            self.props_orthogonal['rotation'] += math.radians(90)
        if self.name == RIDDLE_IDS[9]:
            self.props_orthogonal['pts'][randint(0, 1)] = (self.props['x'], self.props['y'])

    def generate(self, d):
        for p in [self.props, self.props_orthogonal]:
            pts = utils.rotate_points(p['pts'], (p['x'], p['y']), p['rotation'])
            d.line(pts, fill=self.color, width=p['width'])

    def generate_oddity(self, d):
        self.generate(d)

class RightAngleTriangleGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_random_edge_length(self):
        return utils.get_random_of_percentage(0.25, 0.5, self.img_size)

    def set_state(self):
        self.a = self.get_random_edge_length()
        self.b = self.get_random_edge_length()
        self.c = math.sqrt(math.pow(self.a, 2) + math.pow(self.b, 2))
        self.rotation = utils.get_random_rotation()

    def generate(self, d):
        mid = self.img_size // 2
        second_x = self.a // 2 if randint(0, 1) else -self.a // 2
        pts = [(mid - self.a // 2, mid + self.b // 2), (mid + second_x, mid - self.b // 2), (mid + self.a // 2, mid + self.b // 2)]
        pts = utils.rotate_points(pts, (mid, mid), self.rotation)
        min_x_offset, min_y_offset, max_x_offset, max_y_offset = utils.get_minmax_offsets(pts, self.img_size)
        x = randint(-int(min_x_offset), int(max_x_offset))
        y = randint(-int(min_y_offset), int(max_y_offset))
        pts = np.array(pts) + np.array([x, y])
        d.polygon(pts.flatten().tolist(), outline=0)
        for i in range(-3, 3):
            trans_pts = utils.translate_points(pts, (i, 0)).flatten().tolist()
            d.polygon(trans_pts, outline=0)
            trans_pts = utils.translate_points(pts, (0, i)).flatten().tolist()
            d.polygon(trans_pts, outline=0)

    def is_allowed(self, angle, disallowed):
        for point, diff in disallowed:
            if point + diff > 360 or point - diff < 0:
                if not (angle >= (point + diff) % 360 and angle <= (point + 360 - diff) % 360):
                    return False
            if angle > point - diff and angle < point + diff:
                return False
        return True

    def generate_oddity(self, d):
        radius = utils.get_random_of_percentage(0.2, 0.4, self.img_size)
        first_angle = randint(0, 360)
        min_edge_length = 0.2 * self.img_size
        # cosine rule: c^2 = a^2 + b^2 - 2ab*cos(C)
        min_angle_diff = int(math.degrees(math.acos(1 - (min_edge_length) ** 2 / (2 * radius ** 2))))
        right_angle = (first_angle + 180) % 360
        right_angle_diff = 30
        disallowed = [(right_angle, right_angle_diff), (first_angle, min_angle_diff)]
        second_angle = randint(0, 360)
        while not self.is_allowed(second_angle, disallowed):
            second_angle = randint(0, 360)
        disallowed.extend([(second_angle, min_angle_diff), ((second_angle + 180) % 360, right_angle_diff)])
        third_angle = randint(0, 360)
        while not self.is_allowed(third_angle, disallowed):
            third_angle = randint(0, 360)
        mid = self.img_size // 2
        pts = [ (mid + radius * math.cos(math.radians(first_angle)),
                 mid + radius * math.sin(math.radians(first_angle))),
                (mid + radius * math.cos(math.radians(second_angle)),
                mid + radius * math.sin(math.radians(second_angle))),
                (mid + radius * math.cos(math.radians(third_angle)),
                mid + radius * math.sin(math.radians(third_angle))), ]
        x_offset = randint(int(0.1 * self.img_size) + radius, int(0.9 * self.img_size) - radius) - mid
        y_offset = randint(int(0.1 * self.img_size) + radius, int(0.9 * self.img_size) - radius) - mid
        pts = utils.translate_points(pts, (x_offset, y_offset))
        d.polygon(pts.flatten().tolist(), outline=self.color)
        for i in range(-3, 3):
            trans_pts = utils.translate_points(pts, (i, 0)).flatten().tolist()
            d.polygon(trans_pts, outline=0)
            trans_pts = utils.translate_points(pts, (0, i)).flatten().tolist()
            d.polygon(trans_pts, outline=0)

class RotationGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_state(self):
        self.point = (utils.get_random_of_percentage(0.1, 0.25, self.img_size),
            self.img_size - utils.get_random_of_percentage(0.1, 0.25, self.img_size))
        offset = (utils.get_random_of_percentage(0, 0.1, self.img_size), utils.get_random_of_percentage(0.3, 0.4, self.img_size))
        while True:
            pts = utils.random_points(randint(3, 4), self.img_size * 0.4, offset=0)
            min_area = (self.img_size * 0.1) ** 2
            max_area = (self.img_size * 0.3) ** 2
            center = np.mean(pts, axis=0)
            pts = utils.sort_clockwise(center, pts)
            area = utils.calc_area(pts)
            if area <= max_area and area >= min_area:
                break
        self.width = utils.get_random_of_percentage(0.01, 0.03, self.img_size)
        new_center = (self.point[0] + offset[0], self.point[1] - offset[1])
        self.line_pts = [self.point, new_center]
        poly_offset = (new_center[0] - center[0], new_center[1] - center[1])
        self.pts = utils.translate_points(pts, poly_offset)
        rotation = math.radians(uniform(35, 70))
        self.pts_2 = utils.rotate_points(self.pts, self.point, rotation)
        new_center = utils.rotate(self.point, new_center, rotation)
        self.line_pts_2 = (self.point, new_center)
        if self.is_oddity:
            rotation = math.radians(uniform(15, 345))
            self.pts_2 = utils.rotate_points(self.pts_2, new_center, rotation)

    def set_state(self):
        while True:
            self.generate_state()
            s_poly_1 = Polygon(self.pts)
            s_poly_2 = Polygon(self.pts_2)
            if not s_poly_1.intersects(s_poly_2) and \
                all([not utils.is_outside_image(pnt, self.img_size, 0) for pnt in self.pts]) and \
                all([not utils.is_outside_image(pnt, self.img_size, 0) for pnt in self.pts_2]):
                break
        self.pts = self.pts.flatten().tolist()

    def generate(self, d):
        point_circle = utils.create_circle_box(self.point, self.width)
        d.ellipse(point_circle, fill=self.color, outline=self.color)
        d.line(self.line_pts)
        d.line(self.line_pts_2)
        d.polygon(self.pts, fill=self.color, outline=self.color)
        d.polygon(self.pts_2, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class SquareGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_state(self):
        self.x, self.y, self.width, self.height = utils.get_random_rectangle_props(
            self.args, square=(not self.is_oddity))
        self.rotation = utils.get_random_rotation()

    def generate(self, d):
        pts = [(self.x, self.y), (self.x + self.width, self.y), (self.x + self.width, self.y + self.height), (self.x, self.y + self.height)]
        origin_of_rotation = (self.x + self.width // 2, self.y + self.height // 2)
        pts = utils.rotate_points(pts, origin_of_rotation, self.rotation)
        d.polygon(pts, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class SymmetricalGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve_for_a_b(self, height, length):
        # a * l^2 + b * l = 0
        # a * l^2 + 2 * b * l = -4 * h
        A = np.array([[length * length, length], [length * length, 2 * length]])
        B = np.array([[0], [-4 * height]])
        solution = np.linalg.inv(A) @ B
        return solution[0, 0], solution[1, 0]

    def get_points(self, i, negative=False):
        length = int(self.positions[i + 1] * self.length - self.positions[i] * self.length)
        pts = []
        num_pts = length
        mid = self.img_size // 2
        a, b = self.solve_for_a_b(self.heights[i], length)
        for x in range(num_pts + 1):
            multiplier = -1 if negative else 1
            pts.append((x + mid - self.length // 2 + self.positions[i] * self.length, multiplier * (a * x * x + b * x) + mid))
        return pts

    def get_heights(self):
        min_value = self.img_size * 0.05
        max_value = self.img_size * 0.35
        return np.random.uniform(low=min_value, high=max_value, size=(4,))

    def get_positions(self):
        half = uniform(0.4, 0.6)
        first_quarter = half / 2
        third_quarter = (1 + half) / 2
        first_quarter = uniform(first_quarter - 0.1, first_quarter + 0.1)
        third_quarter = uniform(third_quarter - 0.1, third_quarter + 0.1)
        positions = [0, first_quarter, half, third_quarter, 1]
        return positions

    def set_state(self):
        self.length = utils.get_random_of_percentage(0.5, 0.75, self.img_size)
        self.heights = self.get_heights()
        self.positions = self.get_positions()
        self.rotation = math.radians(randint(15, 75))

    def maybe_rotate(self, pts):
        mid = self.img_size // 2
        if self.name == RIDDLE_IDS[19]:
            pts = utils.rotate_points(pts, (mid, mid), math.pi / 2)
        elif self.name == RIDDLE_IDS[21]:
            pts = utils.rotate_points(pts, (mid, mid), self.rotation)
        return pts

    def generate(self, d):
        for i in range(4):
            pts = self.get_points(i)
            pts = self.maybe_rotate(pts)
            d.polygon(pts, fill=self.color)
        for i in range(4):
            pts = self.get_points(i, negative=True)
            pts = self.maybe_rotate(pts)
            d.polygon(pts, fill=self.color)

    def generate_oddity(self, d):
        for i in range(4):
            pts = self.get_points(i)
            pts = self.maybe_rotate(pts)
            d.polygon(pts, fill=self.color)
        while True:
            new_heights = self.get_heights()
            diff = np.abs(self.heights - new_heights)
            total_diff = np.sum(diff)
            min_height_diff = self.img_size * 0.15
            if total_diff < min_height_diff:
                continue
            self.heights = new_heights
            self.positions = self.get_positions()
            break
        for i in range(4):
            pts = self.get_points(i, negative=True)
            pts = self.maybe_rotate(pts)
            d.polygon(pts, fill=self.color)

class SymmetryGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_adjustment_translation(self, pts):
        bbox = utils.get_bbox(pts)
        t_x = 0
        t_y = 0
        noise_x = uniform(0, self.img_size * 0.1)
        noise_y = uniform(0, self.img_size * 0.1)
        if bbox[0] < self.img_size * 0.1 + noise_x:
            t_x = self.img_size * 0.1 + noise_x - bbox[0]
        if bbox[1] < self.img_size * 0.1 + noise_y:
            t_y = self.img_size * 0.1 + noise_y - bbox[1]
        return t_x, t_y

    def set_state(self):
        mid = self.img_size / 2
        self.line_pts = [(0.1 * self.img_size, mid), (0.9 * self.img_size, mid)]
        self.width = utils.get_random_of_percentage(0.01, 0.03, self.args.img_size)
        while True:
            pts = []
            while len(pts) < 3:
                pnt = (uniform(0.1 * self.img_size, 0.9 * self.img_size), uniform(mid + self.img_size * 0.05, self.img_size * 0.9))
                pts.append(pnt)
            if utils.calc_triangle_area(pts) >= (self.img_size * 0.2) ** 2 / 2:
                break
        if self.is_oddity:
            pts_2 = utils.translate_points(pts, -mid, axis=1)
        else:
            pts_2 = np.array(pts)
            pts_2[:, 1] = np.ones((3,)) * self.img_size - pts_2[:, 1]
        if self.name == RIDDLE_IDS[37]:
            self.pts = utils.rotate_points(pts, (mid, mid), math.pi / 2)
            self.pts_2 = utils.rotate_points(pts_2, (mid, mid), math.pi / 2)
            self.line_pts = utils.rotate_points(self.line_pts, (mid, mid), math.pi / 2)
        elif self.name == RIDDLE_IDS[38]:
            rotation = randint(15, 75)
            rotation = abs(180 * randint(0, 1) - rotation)
            self.pts = utils.rotate_points(pts, (mid, mid), math.radians(rotation))
            self.pts_2 = utils.rotate_points(pts_2, (mid, mid), math.radians(rotation))
            self.line_pts = utils.rotate_points(self.line_pts, (mid, mid), math.radians(rotation))
            self.adjust_canvas()
        else:
            self.pts = pts
            self.pts_2 = pts_2.flatten().tolist()

    def adjust_canvas(self):
        offset = self.get_adjustment_translation(self.pts)
        offset_2 = self.get_adjustment_translation(self.pts_2)
        offset = (max(offset[0], offset_2[0]), max(offset[1], offset_2[1]))
        self.pts = utils.translate_points(self.pts, offset)
        self.pts_2 = utils.translate_points(self.pts_2, offset)
        self.line_pts = utils.translate_points(self.line_pts, offset)
        max_value = max(max(utils.get_bbox(self.pts)), max(utils.get_bbox(self.pts_2)), max(utils.get_bbox(self.line_pts)))
        factor = max_value / (self.img_size * 0.9)
        self.pts = self.pts / factor
        self.pts_2 = self.pts_2 / factor
        self.line_pts = self.line_pts / factor
        self.pts = self.pts.flatten().tolist()
        self.pts_2 = self.pts_2.flatten().tolist()
        self.line_pts = self.line_pts.flatten().tolist()

    def generate(self, d):
        d.line(self.line_pts, fill=self.color, width=self.width)
        d.polygon(self.pts, fill=self.color, outline=self.color)
        d.polygon(self.pts_2, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class TranslationGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_adjustment_translation(self, pts):
        bbox = utils.get_bbox(pts)
        t_x = 0
        t_y = 0
        noise_x = uniform(0, self.img_size * 0.1)
        noise_y = uniform(0, self.img_size * 0.1)
        if bbox[0] < self.img_size * 0.1 + noise_x:
            t_x = self.img_size * 0.1 + noise_x - bbox[0]
        if bbox[1] < self.img_size * 0.1 + noise_y:
            t_y = self.img_size * 0.1 + noise_y - bbox[1]
        return t_x, t_y

    def is_valid_rotation(self, pts, pts_rotated):
        # checks if the rotated triangle is too similar to the original
        # if there is at least one point in the rotated triangle a certain
        # distance away from every point in the original then it is not too similar
        valid = False
        for pnt in pts_rotated:
            pnt = Point(pnt)
            valid = valid or all([pnt.distance(p) >= 0.05 * self.img_size for p in pts])
        return valid

    def set_state(self):
        while True:
            pts = utils.random_points(3, self.img_size)
            area = utils.calc_triangle_area(pts)
            if not isinstance(area, complex) and area >= (self.img_size * 0.4) ** 2 / 2:
                break
        center = ((pts[0][0] + pts[1][0] + pts[2][0]) / 3, (pts[0][1] + pts[1][1] + pts[2][1]) / 3)
        pts_2 = pts[:]
        if self.is_oddity:
            min_rotation = 15
            while True:
                rotation = uniform(min_rotation, 360 - min_rotation)
                pts_2 = utils.rotate_points(pts_2, center, math.radians(rotation))
                if self.is_valid_rotation(pts, pts_2):
                    break
        bbox_1_w, bbox_1_h = utils.get_bbox_lengths(pts)
        bbox_2_w, bbox_2_h = utils.get_bbox_lengths(pts_2)
        diff = self.img_size * 0.05
        min_x_translation = bbox_1_w / 2 + bbox_2_w / 2 + diff
        min_y_translation = bbox_1_h / 2 + bbox_2_h / 2 + diff
        max_x_translation = min_x_translation + self.img_size * 0.3
        max_y_translation = min_y_translation + self.img_size * 0.3
        while True:
            translation = (uniform(-max_x_translation, max_x_translation), uniform(-max_y_translation, max_y_translation))
            if abs(translation[0]) >= min_x_translation or abs(translation[1]) >= min_y_translation:
                break
        pts_2 = utils.translate_points(pts_2, translation)
        translation = self.get_adjustment_translation(pts_2)
        pts = utils.translate_points(pts, translation)
        pts_2 = utils.translate_points(pts_2, translation)
        max_value = max(max(utils.get_bbox(pts)), max(utils.get_bbox(pts_2)))
        factor = max_value / (self.img_size * 0.9)
        pts = pts / factor
        pts_2 = pts_2 / factor
        self.pts = pts.flatten().tolist()
        self.pts_2 = pts_2.flatten().tolist()
        
    def generate(self, d):
        d.polygon(self.pts, fill=self.color, outline=self.color)
        d.polygon(self.pts_2, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        self.generate(d)

class TrapezoidGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, d):
        args = self.args
        props = utils.get_random_line_props(args, offset=0.05)
        height = utils.get_random_of_percentage(0.2, 0.42, args.img_size)  # 0.42 because diagonal ~= 0.6
        length_top = utils.get_random_of_percentage(0.2, 0.42, args.img_size)
        length_bottom = utils.get_random_of_percentage(0.2, 0.42, args.img_size)
        top_left = (props['x'] - length_top // 2, props['y'] - height // 2)
        top_right = (props['x'] + length_top // 2, props['y'] - height // 2)
        bottom_right = (props['x'] + length_bottom // 2, props['y'] + height // 2)
        bottom_left = (props['x'] - length_bottom // 2, props['y'] + height // 2)
        pts = [top_left, top_right, bottom_right, bottom_left]
        rotated_pts = utils.rotate_points(pts, (props['x'], props['y']), props['rotation'])
        upper_line = Line(rotated_pts[0], rotated_pts[1])
        bottom_line = Line(rotated_pts[2], rotated_pts[3])
        bounding_lines = utils.get_bounding_lines(args)
        max_upper_left_offset = args.img_size
        max_upper_right_offset = args.img_size
        max_lower_left_offset = args.img_size
        max_lower_right_offset = args.img_size
        for b_line in bounding_lines:
            upper_intersection = upper_line.intersection(b_line)
            lower_intersection = bottom_line.intersection(b_line)
            if upper_intersection:
                max_upper_left_offset = min(Point(rotated_pts[0]).distance(upper_intersection), max_upper_left_offset)
                max_upper_right_offset = min(Point(rotated_pts[1]).distance(upper_intersection), max_upper_right_offset)
            if lower_intersection:
                max_lower_right_offset = min(Point(rotated_pts[2]).distance(lower_intersection), max_lower_right_offset)
                max_lower_left_offset = min(Point(rotated_pts[3]).distance(lower_intersection), max_lower_left_offset)
        top_offset = randint(-int(max_upper_left_offset), int(max_upper_right_offset))
        bottom_offset = randint(-int(max_lower_left_offset), int(max_lower_right_offset))
        top_left = (pts[0][0] + top_offset, pts[0][1])
        top_right = (pts[1][0] + top_offset, pts[1][1])
        bottom_right = (pts[2][0] + bottom_offset, pts[2][1])
        bottom_left = (pts[3][0] + bottom_offset, pts[3][1])
        pts = [top_left, top_right, bottom_right, bottom_left]
        pts = utils.rotate_points(pts, (props['x'], props['y']), props['rotation'])
        d.polygon(pts, fill=self.color, outline=self.color)

    def generate_oddity(self, d):
        x = utils.get_random_of_percentage(0.4, 0.6, self.img_size)
        y = utils.get_random_of_percentage(0.4, 0.6, self.img_size)
        no_parallel = False
        while not no_parallel:
            pts = []
            slopes = []
            for corner in range(4):
                x_offset = utils.get_random_of_percentage(0.15, 0.3, self.img_size)
                y_offset = utils.get_random_of_percentage(0.15, 0.3, self.img_size)
                x_multiplier = 1 if corner == 1 or corner == 2 else -1
                y_multiplier = 1 if corner == 2 or corner == 3 else -1
                pts.append((x + x_multiplier * x_offset, y + y_multiplier * y_offset))
            no_parallel = True
            for pnt_i in range(4):
                slope = Line(pts[pnt_i], pts[(pnt_i + 1) % 4]).slope()
                for old_slope in slopes:
                    if abs(old_slope - slope) < MIN_SLOPE_DIFF:  # higher slope diffs cause more retries
                        no_parallel = False
                slopes.append(slope)
        d.polygon(pts, fill=self.color, outline=self.color)
