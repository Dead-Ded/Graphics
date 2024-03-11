import numpy as np
from PIL import Image, ImageOps
import math


class Vertex:

    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z


class Model:

    def __init__(self, scale=1, offset=0):
        self.scale, self.offset = scale, offset
        self.points = list()
        self.polygons = list()

    def parse(self, f):
        l = f.readlines()
        for s in l:
            match s[0:2]:
                case "v ":
                    self.points.append(tuple(map(lambda x: float(x) * self.scale + self.offset, s.split()[1:])))
                    # Redo scale and offset
                case "f ":
                    self.polygons.append(tuple(map(
                        lambda x: self.points[int(x.split('/')[0]) - 1], s.split()[1:])
                    ))
            # print(self.polygons)

    def dot_render(self):
        global H, W
        mat = np.zeros((H, W), dtype=np.uint8)
        for p in self.points:
            mat[round(p[1])][round(p[0])] = 255
            mat[round(p[1])][round(p[0]) + 1] = 255
            mat[round(p[1])][round(p[0]) - 1] = 255
            mat[round(p[1]) + 1][round(p[0])] = 255
            mat[round(p[1]) - 1][round(p[0])] = 255
        return Image.fromarray(mat)

    def render(self):
        global H, W
        mat = np.zeros((H, W), dtype=np.uint8)
        for p in self.polygons:
            for j in range(len(p)):
                line(p[j][0], p[j][1], p[(j + 1) % len(p)][0], p[(j + 1) % len(p)][1], 255, mat)
        return Image.fromarray(mat)

    def render_bres(self):
        global H, W
        mat = np.zeros((H, W), dtype=np.uint8)
        for p in self.polygons:
            for j in range(len(p)):
                line_bresenhem(p[j][0], p[j][1], p[(j + 1) % len(p)][0], p[(j + 1) % len(p)][1], 255, mat)
        return Image.fromarray(mat)


def line(x0, y0, x1, y1, colour, image: np.array(np.array(0, dtype=float)), scale=1, offset=0):
    t = None
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0
    for x in range(round(x0) * scale + offset, round(x1) * scale + offset):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        # x, y = x * scale + offset, y * scale + offset
        if xchange:
            image[x, y] = colour
        else:
            image[y, x] = colour


def line_bresenhem(x0, y0, x1, y1, colour, image: np.array(np.array(0, dtype=float))):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0
    y = round(y0)
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(round(x0), round(x1)):
        ...
        derror += dy
        if derror > (x1 - x0):
            derror -= 2 * (x1 - x0)
            y += y_update
        if xchange:
            image[x, y] = colour
        else:
            image[y, x] = colour


H, W = 4000, 4000

model = Model()  # offset, scale
# with open("12221_Cat_v1_l3.obj") as file:
with open("model_2.obj") as file:
    model.parse(file)
# print(model.polygons)
# print(model.points)
# img = model.render()
img = model.dot_render()
img = ImageOps.flip(img)
img.show("lab1_dots.png")
img = model.render_bres()
img = ImageOps.flip(img)
img.show("lab1_render.png")
