# Copyright (c) 2023 IBM Research. All Rights Reserved.
#
# Code accompanying a manuscript entitled:
# "On the visual analytic intelligence of neural networks"

import math
import numpy as np

class Point():
    def __init__(self, x, y=None):
        if y is None and not (isinstance(x, tuple) or isinstance(x, list) or isinstance(x, np.ndarray)):
            raise ValueError('Point must be either list, tuple or 2 values')
        if y is None:
            self.x = x[0]
            self.y = x[1]
        else:
            self.x = x
            self.y = y

    def __getitem__(self, key):
        if key != 0 and key != 1:
            raise ValueError('Point can only have 2 values. Key: {} was used'.format(key))
        return self.x if key == 0 else self.y

    def __iter__(self):
        return iter([self.x, self.y])

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)

    def distance(self, pnt):
        dx = self.x - pnt[0]
        dy = self.y - pnt[1]
        return math.sqrt(dx**2 + dy**2)

class Circle():
    def __init__(self, center, radius):
        if not isinstance(center, Point):
            center = Point(center)
        self.center = center
        self.radius = radius

    def intersection_line(self, line):
        return line.intersection_circle(self)

class Line():
    def __init__(self, p1, p2):
        if not isinstance(p1, Point):
            p1 = Point(p1)
        if not isinstance(p2, Point):
            p2 = Point(p2)
        self.p1 = p1
        self.p2 = p2
        self.points = [p1, p2]
        self.distance = self.p1.distance(self.p2)
        self.A = (p1[1] - p2[1])
        self.B = (p2[0] - p1[0])
        self.C = -(p1[0] * p2[1] - p2[0] * p1[1])

    def slope(self):
        return (self.p2[1] - self.p1[1]) / (self.p2[0] - self.p1[0] + 1e-8)

    def intersection(self, line):
        D = self.A * line.B - self.B * line.A
        Dx = self.C * line.B - self.B * line.C
        Dy = self.A * line.C - self.C * line.A
        if D != 0:
            x = Dx / D
            y = Dy / D
            return Point(x, y)
        return False

    def intersection_circle(self, circle):
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        a = dx * dx + dy * dy
        b = 2 * (dx * (self.p1.x - circle.center.x) + dy * (self.p1.y - circle.center.y))
        c = self.p1.x * self.p1.x + self.p1.y * self.p1.y
        c += circle.center.x * circle.center.x + circle.center.y * circle.center.y
        c -= 2 * (self.p1.x * circle.center.x + self.p1.y * circle.center.y)
        c -= circle.radius * circle.radius
        det = b * b - 4 * a * c
        if det < 0:
            return False
        t_minus = (-b - math.sqrt(det)) / (2 * a)
        t_plus = (-b + math.sqrt(det)) / (2 * a)
        if (t_minus >= 0 and t_minus <= 1) or (t_plus >= 0 and t_plus <= 1):
            return Point(self.p1.x + t_minus * dx, self.p1.y + t_minus * dy), Point(self.p1.x + t_plus * dx, self.p1.y + t_plus * dy)
        return False
