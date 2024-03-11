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
                case "f ":
                    self.polygons.append(tuple(map(lambda x: self.points[int(
                        x.split('/')[0]
                    ) - 1], s.split()[1:])))
            # print(self.polygons)

    def render(self):
        mat = np.zeros((4000, 4000), dtype=np.uint8)
        for p in self.polygons:
            for j in range(len(p)):
                line(p[j][0], p[j][1], p[(j + 1) % len(p)][0], p[(j + 1) % len(p)][1], 255, mat)
        return Image.fromarray(mat)

    def render_bres(self):
        mat = np.zeros((4000, 4000), dtype=np.uint8)
        for p in self.polygons:
            for j in range(len(p)):
                line_bresenhem(p[j][0], p[j][1], p[(j + 1) % len(p)][0], p[(j + 1) % len(p)][1], 255, mat)
        return Image.fromarray(mat)


def dotted_line(x0, y0, x1, y1, colour, count, image):
    step = 1.0 / count
    t = None
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = colour


def dotted_line2(x0, y0, x1, y1,
                 colour, image):
    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = colour


# def x_loop_line(image, x0, y0, x1, y1, color):
# for x in range(x0, x1):
#     t = (x - x0) / (x1 - x0)
#     y = round((1.0 - t) * y0 + t * y1)
#     image[y, x] = color

def line_by_x(x0, y0, x1, y1, colour, image):
    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(round(x0), round(x1)):
        ...
        image[y, x] = colour
        derror += dy
        if derror > 0.5:
            derror -= 1.0
            y += y_update


def line2x(x0, y0, x1, y1, colour, image):
    y = y0
    dy = 2.0 * (x1 - x0) * abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(round(x0), round(x1)):
        ...
        image[y, x] = colour
        derror += dy
        if derror > 2.0 * (x1 - x0) * 0.5:
            derror -= 2.0 * (x1 - x0) * 1.0
            y += y_update


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


H, W = 800, 800

# 1
mat = np.zeros((H, W), dtype=np.uint8)
img = Image.fromarray(mat)
img.save("lab1_1.png")

# 2
mat = np.array([[255] * W] * H, dtype=np.uint8)
img2 = Image.fromarray(mat)
img2.save("lab1_2.png")

# 3
mat = np.array([[(255, 0, 0)] * W] * H, dtype=np.uint8)
img2 = Image.fromarray(mat)
img2.save("lab1_3.png")

# 4
mat = np.array([[(x + y) % 256 for x in range(W)] for y in range(H)], dtype=np.uint8)
img2 = Image.fromarray(mat)
img2.save("lab1_4.png")

h, w = 200, 200

mat = np.zeros((h, w), dtype=np.uint8)
for i in range(13):
    dotted_line(100, 100, 100 + 95 * math.cos(2 * math.pi * i / 13), 100 + 95 * math.sin(2 * math.pi * i / 13), 255,
                100, mat)
img = Image.fromarray(mat)
img.save("lab1_5.png")

mat = np.zeros((h, w), dtype=np.uint8)
for i in range(13):
    dotted_line2(100, 100, 100 + 95 * math.cos(2 * math.pi * i / 13), 100 + 95 * math.sin(2 * math.pi * i / 13), 255,
                 mat)
img = Image.fromarray(mat)
img.save("lab1_6.png")

mat = np.zeros((h, w), dtype=np.uint8)
for i in range(13):
    line_by_x(100, 100, 100 + 95 * math.cos(2 * math.pi * i / 13), 100 + 95 * math.sin(2 * math.pi * i / 13), 255,
              mat)
img = Image.fromarray(mat)
img.save("lab1_7.png")

mat = np.zeros((h, w), dtype=np.uint8)
for i in range(13):
    line2x(100, 100, 100 + 95 * math.cos(2 * math.pi * i / 13), 100 + 95 * math.sin(2 * math.pi * i / 13), 255,
           mat)
img = Image.fromarray(mat)
img.save("lab1_8.png")

mat = np.zeros((h, w), dtype=np.uint8)
for i in range(13):
    line(100, 100, 100 + 95 * math.cos(2 * math.pi * i / 13), 100 + 95 * math.sin(2 * math.pi * i / 13), 255,
         mat)
img = Image.fromarray(mat)
img.save("lab1_9.png")

mat = np.zeros((h, w), dtype=np.uint8)
for i in range(13):
    line_bresenhem(100, 100, 100 + 95 * math.cos(2 * math.pi * i / 13), 100 + 95 * math.sin(2 * math.pi * i / 13), 255,
                   mat)
img = Image.fromarray(mat)
img.save("lab1_10.png")

model = Model(25000, 1500)
# with open("12221_Cat_v1_l3.obj") as file:
with open("model_1.obj") as file:
    model.parse(file)
# print(model.polygons)
img = model.render()
img = ImageOps.flip(img)
img.save("lab1_11.png")

model = Model(25000, 1500)
# with open("12221_Cat_v1_l3.obj") as file:
with open("model_1.obj") as file:
    model.parse(file)
# print(model.polygons)
img = model.render_bres()
img = ImageOps.flip(img)
img.save("lab1_12.png")
