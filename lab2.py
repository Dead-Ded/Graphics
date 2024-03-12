from lab1 import *
from random import randint
from math import sqrt


class Model(Model):
    def __init__(self, height=0, width=0, scale=1, offset=0):
        super().__init__(height, width, scale, offset)
        self.normals = list()

    def __bar_coords(self, point=(0, 0, 0), polygon=((0, 0, 0), (0, 0, 0), (0, 0, 0))):
        x, y = point[0], point[1]
        x0, y0, x1, y1, x2, y2 = polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1], polygon[2][0], polygon[2][
            1]
        l0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / \
             ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
        l1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / \
             ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
        l2 = 1 - l0 - l1
        return l0, l1, l2

    def __square(self, polygon=((0, 0, 0), (0, 0, 0), (0, 0, 0))):
        min_x, min_y = min(polygon[0][0], polygon[1][0], polygon[2][0]), min(polygon[0][1], polygon[1][1],
                                                                             polygon[2][1])
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = max(polygon[0][0], polygon[1][0], polygon[2][0]), max(polygon[0][1], polygon[1][1],
                                                                             polygon[2][1])
        max_x, max_y = min(self.width, max_x), min(self.height, max_y)
        return tuple(map(round, (min_x, min_y, max_x, max_y)))

    def calc_normals(self):
        for polygon in self.polygons:
            x0, y0, z0, x1, y1, z1, x2, y2, z2 = (
                polygon[0][0], polygon[0][1], polygon[0][2], polygon[1][0], polygon[1][1], polygon[1][2], polygon[2][0],
                polygon[2][1], polygon[2][2]
            )
            self.normals.append(
                (
                    (y1 - y2) * (z1 - z0) - (y1 - y0) * (z1 - z2),
                    (x1 - x2) * (z1 - z0) - (x1 - x0) * (z1 - z2),
                    (x1 - x2) * (y1 - y0) - (x1 - x0) * (y1 - y2)
                 )
            )

    def render_old(self):
        mat = np.array([[(0, 0, 0)] * H] * W, dtype=np.uint8)
        for p in self.polygons:
            colour = (randint(0, 255), randint(0, 255), randint(0, 255))
            square = self.__square(p)
            for i in range(square[1], square[3] + 1):
                for j in range(square[0], square[2] + 1):
                    bar_coords = self.__bar_coords((j, i), p)
                    if all(map(lambda x: x >= 0, bar_coords)):
                        mat[i][j] = colour
        return Image.fromarray(mat)

    def render(self):
        l = (0, 0, 1)
        mat = np.array([[(0, 0, 0)] * H] * W, dtype=np.uint8)
        self.calc_normals()
        for k in range(len(self.polygons)):
            normal = self.normals[k]
            scal = normal[0] * l[0] + normal[1] * l[1] + normal[2] * l[2]
            if scal >= 0:
                continue
            p = self.polygons[k]
            colour = (0, - 255 * scal // int(sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)), 0)
            square = self.__square(p)
            for i in range(square[1], square[3] + 1):
                for j in range(square[0], square[2] + 1):
                    bar_coords = self.__bar_coords((j, i), p)
                    if all(map(lambda x: x >= 0, bar_coords)):
                        mat[i][j] = colour
        return Image.fromarray(mat)


if __name__ == "__main__":
    H, W = 4000, 4000
    model = Model(H, W, 20000, 2000)
    with open("model_1.obj") as file:
        model.parse(file)
    img = model.render()
    img = ImageOps.flip(img)
    img.show("lab2_triangles.png")
