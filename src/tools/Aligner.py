import math
import warnings
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk, ImageDraw
import argparse
import json
import os
import msgpack
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import scipy.spatial.distance
import scipy.special
import collections
import natsort
from ttkbootstrap import Style
from tqdm import tqdm
import struct


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', default=None, required=True,
                        help='path to maps')
    parser.add_argument('--src_dir', default=None, required=True,
                        help='path to src_dir')
    parser.add_argument('--outf', default=None, required=True,
                        help='path to outf')
    parser.add_argument('--plan', default=None, required=True,
                        help='path to plan')
    parser.add_argument('--file_name', default=None, required=True,
                        help='name of json file')
    opt = parser.parse_args()
    return opt


def load_data(opt):
    if not os.path.exists(os.path.join(opt.outf, 'data.json')):
        if not os.path.exists(opt.outf):
            os.makedirs(opt.outf)
        if opt.maps.endswith('.msg'):
            ffid, features, kf = get_slam_data(opt)
        else:
            loader = get_colmap_data(opt)
            ffid, features, kf = loader.load()
        with open(os.path.join(opt.outf, 'data.json'), 'w') as f:
            json.dump({'ffid': ffid, 'features': features, 'kf': kf}, f)
    else:
        with open(os.path.join(opt.outf, 'data.json'), 'r') as f:
            t = json.load(f)
        ffid = t['ffid']
        features = t['features']
        kf = t['kf']
    features = np.array(features)
    kf = np.array(kf)
    return ffid, features, kf


class Thin_Plate_Spline():
    def __init__(self, src_pts, des_pts):
        self.control_points = src_pts
        self.L = self._L(src_pts)
        self.coeffs = self._coeffs(self.L, des_pts)

    def _coeffs(self, L, des_pts):
        V = np.vstack((des_pts, np.zeros((3, 2))))
        coeffs = np.linalg.solve(L, V)
        return coeffs

    def _K(self, pts):
        pts = np.atleast_2d(pts)
        r = scipy.spatial.distance.cdist(pts, pts)
        return scipy.special.xlogy(r ** 2, r)

    def _L(self, src_pts):
        n = src_pts.shape[0]
        P = np.hstack((np.ones((n, 1)), src_pts))
        L = np.zeros((n + 3, n + 3))
        L[0:n, 0:n] = self._K(src_pts)
        L[0:n, n:n + 3] = P
        L[n:n + 3, 0:n] = P.T
        return L

    def _U(self, x):
        return (x ** 2) * np.where(x < 1e-100, 0, np.log(x))

    def wrap_points(self, pts):
        wrap_pts = np.empty_like(pts)
        for k in range(len(pts)):
            for j in range(2):
                wrap_pts[k, j] = self.coeffs[-3, j] + self.coeffs[-2, j] * pts[k, 0] + self.coeffs[-1, j] * pts[k, 1]
                for i in range(len(self.coeffs) - 3):
                    r2 = ((self.control_points[i, 0] - pts[k, 0]) ** 2.0 + (
                                self.control_points[i, 1] - pts[k, 1]) ** 2.0) ** 0.5
                    if r2 != 0:
                        wrap_pts[k, j] += self.coeffs[i, j] * scipy.special.xlogy(r2 ** 2, r2)
        return wrap_pts


class get_colmap_data():
    def __init__(self, opt):
        self.images_path = os.path.join(opt.maps, 'images.bin')
        self.point3d_path = os.path.join(opt.maps, 'points3D.bin')

    def read_next_bytes(self, fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    def read_points3d_binary(self, path_to_model_file):
        points3D = {}
        with open(path_to_model_file, "rb") as fid:
            num_points = self.read_next_bytes(fid, 8, "Q")[0]
            for point_line_index in range(num_points):
                binary_point_line_properties = self.read_next_bytes(
                    fid, num_bytes=43, format_char_sequence="QdddBBBd")
                point3D_id = binary_point_line_properties[0]
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = self.read_next_bytes(
                    fid, num_bytes=8, format_char_sequence="Q")[0]
                track_elems = self.read_next_bytes(
                    fid, num_bytes=8 * track_length,
                    format_char_sequence="ii" * track_length)
                image_ids = np.array(tuple(map(int, track_elems[0::2])))
                point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
                points3D[point3D_id] = {'id': point3D_id, 'xyz': xyz, 'rgb': rgb,
                                        'error': error, 'image_ids': image_ids,
                                        'point2D_idxs': point2D_idxs}
        return points3D

    def read_images_binary(self, path_to_model_file):
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = self.read_next_bytes(fid, 8, "Q")[0]
            for image_index in range(num_reg_images):
                binary_image_properties = self.read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = ""
                current_char = self.read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8")
                    current_char = self.read_next_bytes(fid, 1, "c")[0]
                num_points2D = self.read_next_bytes(fid, num_bytes=8,
                                                    format_char_sequence="Q")[0]
                x_y_id_s = self.read_next_bytes(fid, num_bytes=24 * num_points2D,
                                                format_char_sequence="ddq" * num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                       tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                images[image_id] = {'id': image_id, 'qvec': qvec, 'tvec': tvec,
                                    'camera_id': camera_id, 'name': image_name,
                                    'xys': xys, 'point3D_ids': point3D_ids}
        return images

    def load(self):
        images = self.read_images_binary(self.images_path)
        images = collections.OrderedDict(natsort.natsorted(images.items()))
        P3D = self.read_points3d_binary(self.point3d_path)
        ffid = {}
        key = []
        point3d = []
        for id, point in P3D.items():
            pos = point["xyz"]
            point3d.append([pos[0], 0, pos[2]])
        for id, point in images.items():
            trans = point["tvec"]
            rot = point['qvec']
            rot = [rot[1], rot[2], rot[3], rot[0]]
            pos = slam2world(trans, rot)
            key.append([pos[0], pos[2]])
        kf = np.array(key)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(point3d)
        cl, source1 = source.remove_radius_outlier(nb_points=16, radius=0.25)
        source1 = source.select_by_index(source1)
        features = np.array(source1.points)
        features = np.vstack((features[:, 0], features[:, 2])).T
        for i in tqdm(list(images.keys())):
            t = images[i]['name']
            im = opt.src_dir + '/' + t
            jj = images[i]['point3D_ids']
            j = [int(ii) for ii in jj if ii != -1]
            lm = [P3D[i]['xyz'].tolist() for i in j]
            kp = images[i]['xys']
            rot = images[i]['qvec']
            rot = [rot[1], rot[2], rot[3], rot[0]]
            gp = []
            index = [ii for ii, x in enumerate(images[i]['point3D_ids']) if x != -1]
            for ii, tt in enumerate(kp):
                if ii in index:
                    gp.append(np.array(tt))
            ffid.update(
                {t.replace('.png', ''): {'frame': im, 'ang': R.from_quat(rot).as_rotvec()[1], 'lm': lm, 'lm_id': j,
                                         'gp': np.array(gp).tolist()}})
        return ffid, features.tolist(), kf.tolist()


def get_slam_data(opt):
    ffid = {}
    bin_fn = opt.maps
    with open(bin_fn, "rb") as f:
        data = msgpack.unpackb(f.read(), use_list=False, raw=False)
    landmarks = data['landmarks']
    keyframes = collections.OrderedDict(natsort.natsorted(data['keyframes'].items()))
    point3d = []
    key = []
    for id, point in landmarks.items():
        pos = point["pos_w"]
        point3d.append([pos[0], 0, pos[2]])
    for id, point in keyframes.items():
        trans = point["trans_cw"]
        rot = point['rot_cw']
        pos = slam2world(trans, rot)
        key.append([pos[0], pos[2]])
    kf = np.array(key)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point3d)
    cl, source1 = source.remove_radius_outlier(nb_points=16, radius=0.25)
    source1 = source.select_by_index(source1)
    features = np.array(source1.points)
    features = np.vstack((features[:, 0], features[:, 2])).T
    list_dir = os.listdir(opt.src_dir)
    num1 = np.max([len(str(data['keyframes'][i]['src_frm_id'])) for i in data['keyframes']])
    num2 = len(str(list_dir[0].replace('.png', '')))
    num = max(num1, num2)
    for i in tqdm(data['keyframes']):
        t = str(data['keyframes'][i]['src_frm_id'] + 1).zfill(num)
        im = opt.src_dir + '/' + t + '.png'
        jj = data['keyframes'][i]['lm_ids']
        j = [ii for ii in jj if ii != -1]
        lm = [landmarks[str(i)]['pos_w'] for i in j]
        kp = data['keyframes'][i]['keypts']
        rot = data['keyframes'][i]['rot_cw']
        gp = []
        index = [ii for ii, x in enumerate(data['keyframes'][i]['lm_ids']) if x != -1]
        for ii, tt in enumerate(kp):
            if ii in index:
                gp.append(np.array(tt['pt']))
        ffid.update({t: {'frame': im, 'ang': R.from_quat(rot).as_rotvec()[1], 'lm': lm, 'lm_id': j,
                         'gp': np.array(gp).tolist()}})
    for i in list_dir:
        if i.replace('.png', '') not in list(ffid.keys()):
            os.remove(os.path.join(opt.src_dir, i))
    return ffid, features.tolist(), kf.tolist()


def slam2world(t, r):
    r = R.from_quat(r)
    return -np.matmul(r.as_matrix().transpose(), t)


class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)


class Button(ttk.Button):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)


class CanvasImage_FloorPlan:
    """ Display and zoom image """

    def __init__(self, placeholder, parent, data, plan):
        """ Initialize the ImageFrame """
        self.parent = parent
        self.placeholder = placeholder
        self.placeholder.geometry("+{}+{}".format(self.parent.positionRight, self.parent.positionDown))
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.1  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.path = data['frame']  # path to the image, should be public for outer classes
        self.points = data['gp']
        self.lm = data['lm']
        self.text = None
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<Button-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<Button-3>', self.__coordinates)
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(plan)  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
                self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(plan)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        minsize, maxsize, number = 5, 20, 10
        self.width, self.height = self.__image.size
        self.r = 3
        self.chose = {}
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas
        global num_pairs, points3D
        if num_pairs > 3:
            p3 = self.local2global(points3D[list(points3D.keys())[-1]]['lm'])
            x, y = p3
            bbox = self.canvas.coords(self.container)
            scale = (bbox[2] - bbox[0]) / self.imwidth
            x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale
            x0l, x0r, y0l, y0r = (x_ - self.r) * scale + bbox[0], (x_ + self.r) * scale + bbox[0], (
                    y_ - self.r) * scale + bbox[1], (y_ + self.r) * scale + bbox[1]
            self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='yellow', activefill='purple')
            self.canvas.create_text(x_ * scale + bbox[0], (y_ + 2 * self.r) * scale + bbox[1], fill="darkblue",
                                    font="Times 10 italic bold",
                                    text="Estimated location")

    def local2global(self, pt):
        x = np.array([pt[0], pt[2], 1])
        global T
        return T @ x

    def __coordinates(self, event):
        if len(self.chose) == 1:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            bbox = self.canvas.coords(self.container)
            scale = (bbox[2] - bbox[0]) / self.imwidth
            x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale
            for s in self.chose:
                list = s.split('-')
                x0, y0 = float(list[0]), float(list[1])
                if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r):
                    points2D.append([x0, y0])
                    global Frame, FL
                    Frame, FL = True, False
                    if num_pairs > 2:
                        self.parent.match()
                    if num_pairs == 3:
                        self.parent.show_frame('enter')
                    self.placeholder.destroy()

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)

        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k) + 1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30 * ' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                    (int(x1 / self.__scale), int(y1 / self.__scale),
                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.coords(self.container)
        scale = (bbox[2] - bbox[0]) / self.imwidth
        x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale

        if (x_ < self.imwidth) and (x_ > 0) and (y_ < self.imheight) and (y_ > 0):
            x0l, x0r, y0l, y0r = (x_ - self.r) * scale + bbox[0], (x_ + self.r) * scale + bbox[0], (
                    y_ - self.r) * scale + bbox[1], (y_ + self.r) * scale + bbox[1]
            if len(self.chose.keys()) == 1:
                if (str(x_) + '-' + str(y_)) == list(self.chose.keys())[0]:
                    self.canvas.delete(self.chose[str(x_) + '-' + str(y_)]['index'])
                    self.chose.pop(str(x_) + '-' + str(y_))
            if self.text:
                self.canvas.delete(self.text)
            self.text = self.canvas.create_text(x_ * scale + bbox[0], (y_ + self.r * 2) * scale + bbox[1],
                                                fill="darkblue", font="Times 10 italic bold",
                                                text="Right click to pick")
            self.chose.update({str(x_) + '-' + str(y_): {
                'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='red', activefill='green')}})
        if len(self.chose) > 1:
            for i in list(self.chose.keys())[:-1]:
                self.canvas.delete(self.chose[i]['index'])
                self.chose.pop(i)

        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll', 1, 'unit', event=event)


class CanvasImage_Feature:
    """ Display and zoom image """

    def __init__(self, placeholder, parent, data):
        """ Initialize the ImageFrame """
        self.parent = parent
        self.placeholder = placeholder
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.1  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.path = data['frame']  # path to the image, should be public for outer classes
        self.points = data['gp']
        self.lm = data['lm']
        self.text = None
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<Button-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<Button-3>', self.__coordinates)
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
                self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        minsize, maxsize, number = 5, 20, 10
        self.width, self.height = self.__image.size
        self.r = 3
        self.chose = {}
        self.nochose = {}
        self.picked = []
        global points3D
        for j in points3D.keys():
            if points3D[j]['id'] == (self.path.split('/')[-1]).replace('.png', ''):
                x, y = points3D[j]['frame']
                self.canvas.create_oval(x - self.r * 2, y - self.r * 2, x + self.r * 2, y + self.r * 2, fill='yellow',
                                        activefill='purple')
                self.picked.append([x, y])
        for i, n in enumerate(self.points):
            x0 = n[0]
            y0 = n[1]
            if [x0, y0] not in self.picked:
                self.nochose.update({str(x0) + '-' + str(y0): {
                    'index': self.canvas.create_oval(x0 - self.r, y0 - self.r, x0 + self.r, y0 + self.r, fill='green',
                                                     activefill='red'), 'lm': self.lm[i], 'frame': [x0, y0],
                    'id': (self.path.split('/')[-1]).replace('.png', '')}})
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def __coordinates(self, event):
        if len(self.chose) == 1:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            bbox = self.canvas.coords(self.container)
            scale = (bbox[2] - bbox[0]) / self.imwidth
            x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale
            for s in self.chose.keys():
                list1 = s.split('-')
                x0, y0 = float(list1[0]), float(list1[1])
                if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r * 2):
                    global num_pairs
                    points3D.update({str(num_pairs).zfill(3): {'lm': self.chose[s]['lm'],
                                                               'frame': self.chose[s]['frame'],
                                                               'id': self.chose[s]['id']}})
                    num_pairs += 1
                    global Frame, FL
                    Frame, FL = False, True
                    chose = []
                    with Image.open(os.path.join(self.parent.src, self.parent.value + '.png')) as im1:
                        draw = ImageDraw.Draw(im1)
                        for j in list(points3D.keys())[::-1]:
                            if points3D[j]['id'] == self.parent.value:
                                x, y = points3D[j]['frame']
                                draw.rectangle([(x - 20, y - 20), (x + 20, y + 20)], fill=(255, 255, 0))
                                chose.append([x, y])
                                break
                        for points in self.parent.ffid[self.parent.value]['gp']:
                            x, y = points
                            if [x, y] not in chose:
                                draw.rectangle([(x - 10, y - 10), (x + 10, y + 10)], fill=(0, 255, 0))
                    framew, frameh = im1.size
                    scale = 600 / framew
                    newsize = (600, int(frameh * scale))
                    im1 = im1.resize(newsize)
                    tkimage = ImageTk.PhotoImage(im1)
                    self.parent.myvar = Label(self.parent, image=tkimage)
                    self.parent.myvar.bind('<Double-Button-1>', self.parent.new_window_feature)
                    self.parent.myvar.image = tkimage
                    self.parent.myvar.grid(row=1, column=4, columnspan=1, rowspan=10, sticky="snew")

                    self.placeholder.destroy()

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)

        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k) + 1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30 * ' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                    (int(x1 / self.__scale), int(y1 / self.__scale),
                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.coords(self.container)
        scale = (bbox[2] - bbox[0]) / self.imwidth
        x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale
        for x0, y0 in self.points:
            if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r * 2) and (str(x0) + '-' + str(y0) in self.chose):
                x0l, x0r, y0l, y0r = (x0 - self.r) * scale + bbox[0], (x0 + self.r) * scale + bbox[0], (
                        y0 - self.r) * scale + bbox[1], (y0 + self.r) * scale + bbox[1]
                self.nochose.update({str(x0) + '-' + str(y0): {
                    'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='green', activefill='red'),
                    'lm': self.chose[str(x0) + '-' + str(y0)]['lm'],
                    'frame': self.chose[str(x0) + '-' + str(y0)]['frame'],
                    'id': self.chose[str(x0) + '-' + str(y0)]['id']}})
                self.canvas.delete(self.chose[str(x0) + '-' + str(y0)]['index'])
                self.chose.pop(str(x0) + '-' + str(y0))
                break
            if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r) and (str(x0) + '-' + str(y0) in self.nochose):
                xx, _, yy = self.nochose[str(x0) + '-' + str(y0)]['lm']
                if self.text:
                    self.canvas.delete(self.text)
                x0l, x0r, y0l, y0r = (x0 - self.r * 2) * scale + bbox[0], (x0 + self.r * 2) * scale + bbox[0], (
                        y0 - self.r * 2) * scale + bbox[1], (y0 + self.r * 2) * scale + bbox[1]
                self.text = self.canvas.create_text(x0 * scale + bbox[0], (y0 + self.r * 3) * scale + bbox[1],
                                                    fill="darkblue", font="Times 10 italic bold",
                                                    text="Right click to pick")
                self.chose.update({str(x0) + '-' + str(y0): {
                    'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='red', activefill='green'),
                    'lm': self.nochose[str(x0) + '-' + str(y0)]['lm'],
                    'frame': self.nochose[str(x0) + '-' + str(y0)]['frame'],
                    'id': self.nochose[str(x0) + '-' + str(y0)]['id']}})
                self.canvas.delete(self.nochose[str(x0) + '-' + str(y0)]['index'])
                self.nochose.pop(str(x0) + '-' + str(y0))
                x_, y_ = x0, y0
                if len(self.chose) > 1:
                    for i in list(self.chose.keys())[:-1]:
                        x0, y0 = i.split('-')
                        x0, y0 = float(x0), float(y0)
                        x0l, x0r, y0l, y0r = (x0 - self.r) * scale + bbox[0], (x0 + self.r) * scale + bbox[0], (
                                y0 - self.r) * scale + bbox[1], (y0 + self.r) * scale + bbox[1]
                        self.nochose.update({i: {
                            'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='green', activefill='red'),
                            'lm': self.chose[i]['lm'],
                            'frame': self.chose[i]['frame'],
                            'id': self.chose[i]['id']}})
                        self.canvas.delete(self.chose[i]['index'])
                        self.chose.pop(i)
                global num_pairs
                im1 = self.parent.fm.copy()
                framew, frameh = im1.size
                scale = 600 / framew
                newsize = (600, int(frameh * scale))
                im1 = im1.resize(newsize)
                tkimage = ImageTk.PhotoImage(im1)
                self.parent.myvar = Label(self.parent, image=tkimage)
                self.parent.myvar.bind('<Double-Button-1>', self.parent.new_window_feature)
                self.parent.myvar.image = tkimage
                self.parent.myvar.grid(row=1, column=4, columnspan=1, rowspan=10, sticky="snew")
                if num_pairs > 2:
                    t_fp = np.array([xx, yy, 1]).T
                    t_mp = T @ t_fp
                    im = self.parent.imfc.copy()
                    draw = ImageDraw.Draw(im)
                    x, y = t_mp.T
                    draw.rectangle([(x - 10, y - 10), (x + 10, y + 10)], fill="#ffff33", outline="red")
                    ind = self.parent.dics.index(self.parent.value)
                    t_fp = np.array([self.parent.kf[ind, 0], self.parent.kf[ind, 1], 1]).T
                    t_mp = T @ t_fp
                    ang = self.parent.ffid[self.parent.value]['ang'] - self.parent.rot_base
                    xx, yy = t_mp.T
                    x1, y1 = xx - 70 * np.sin(ang), yy - 70 * np.cos(ang)
                    draw.ellipse((xx - 30, yy - 30, xx + 30, yy + 30), fill=(255, 0, 255))
                    draw.line([(xx, yy), (x1, y1)], fill=(255, 0, 255), width=15)
                    width, height = im.size
                    scale = 970 / width
                    newsize = (970, int(height * scale))
                    im = im.resize(newsize)
                    tkimage1 = ImageTk.PhotoImage(im)
                    self.parent.myvar1 = Label(self.parent, image=tkimage1)
                    self.parent.myvar1.bind('<Double-Button-1>', self.parent.new_window_fl)
                    self.parent.myvar1.image = tkimage1
                    self.parent.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")
                break
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll', 1, 'unit', event=event)


class Features_window(ttk.Frame):
    def __init__(self, mainframe, parent, data, name):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Pick Frame Feature Point (Current Frame: {})'.format(name))
        self.master.geometry('1200x600')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)

        canvas = CanvasImage_Feature(self.master, parent, data)  # create widget
        canvas.grid(row=0, column=0)  # show widget


class FloorPlan_window(ttk.Frame):
    def __init__(self, mainframe, parent, data, name):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title(
            'Pick Floor Plan Land Mark (Floor Plan: {})'.format((name.split('/')[-1]).replace('.png', '')))
        self.master.geometry('1800x900')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)

        canvas = CanvasImage_FloorPlan(self.master, parent, data, name)  # create widget
        canvas.grid(row=0, column=0)  # show widget


class Main_window(ttk.Frame):
    def __init__(self, master, ffid, src, features, kf, outf, opt, file_name):
        ttk.Frame.__init__(self, master=master)
        self.opt = opt
        self.ffid, self.src = ffid, src
        self.style = ttk.Style()
        self.features = features
        self.kf = kf
        self.rot_scale = ((max(kf[:, 0]) - min(kf[:, 0])) ** 2 + (max(kf[:, 1]) - min(kf[:, 1])) ** 2) ** (0.5) / 1000
        self.plan = None
        self.outf = outf
        self.save = False
        self.save_topomap = False
        self.coeffs = None
        self.control_pts = None
        self.file_name = file_name

        windowWidth = self.master.winfo_reqwidth()
        windowHeight = self.master.winfo_reqheight()
        self.positionRight = int(self.master.winfo_screenwidth() / 2 - windowWidth / 2)
        self.positionDown = int(self.master.winfo_screenheight() / 2 - windowHeight / 2)
        self.master.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        global points2D, points3D, Frame, FL, num_pairs, T
        points2D, points3D = [], {}
        num_pairs = 0
        self.points2D, self.points3D = [], {}
        Frame, FL = True, False
        self.master.title('Coordinates Aligner')
        self.pack(side="left", fill="both", expand=False)

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        window_width = int(screen_width * 0.8)  # 80% of screen width
        window_height = int(screen_height * 0.8)  # 80% of screen height
        self.master.geometry(f"{window_width}x{window_height}")

        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(3, pad=7)
        self.master.rowconfigure(3, weight=1)
        self.master.rowconfigure(6, pad=7)
        # --------------------------------------------------
        lbl = Label(self, text="Key frame list:")
        lbl.grid(row=0, column=0, sticky=W, pady=4, ipadx=2)

        self.c = ttk.LabelFrame(self, text='Matching Method')
        self.c.grid(row=1, column=1, columnspan=1, padx=10, sticky=E + W)
        self.v1 = tk.IntVar()
        self.linear_matching = True
        self.lb2 = tk.Radiobutton(self.c,
                                  text='Linear',
                                  command=self.matching_type,
                                  variable=self.v1,
                                  value=0).pack(side=LEFT)
        self.lb2 = tk.Radiobutton(self.c,
                                  text='TPS',
                                  command=self.matching_type,
                                  variable=self.v1,
                                  value=1).pack(side=RIGHT)

        var2 = tk.StringVar()
        self.lb = tk.Listbox(self, listvariable=var2)

        self.scrollbar = Scrollbar(self, orient=VERTICAL)
        self.lb.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.lb.yview)

        self.lb.bind('<Double-Button-1>', lambda event, action='double':
        self.show_frame(action))
        self.lb.bind('<Return>', self.new_window_feature)
        self.lb.bind('<Up>', lambda event, action='up':
        self.show_frame(action))
        self.lb.bind('<Down>', lambda event, action='down':
        self.show_frame(action))
        self.dics = sorted(list(ffid.keys()))

        for i in self.dics:
            self.lb.insert('end', i)

        self.scrollbar.grid(row=1, column=0, columnspan=2, rowspan=9, padx=2, sticky='sn')
        self.lb.grid(row=1, column=0, columnspan=1, rowspan=9, padx=2,
                     sticky=E + W + S + N)

        # ---------------------------------------------------
        separatorh = ttk.Separator(self, orient='horizontal')
        separatorh.grid(row=20, column=0, pady=10, ipadx=1, columnspan=40, rowspan=1, sticky="ew")
        separatorv1 = ttk.Separator(self, orient='vertical')
        separatorv1.grid(row=0, column=3, padx=10, columnspan=1, rowspan=70, sticky="sn")
        separatorv2 = ttk.Separator(self, orient='vertical')
        separatorv2.grid(row=0, column=40, ipadx=1, columnspan=1, rowspan=70, sticky="sn")
        abtn = tk.Button(self, text='Select Floor Plan', width=14, command=self.FloorPlan_select)
        abtn.grid(row=21, padx=10, columnspan=1, column=1)
        self.style.layout(
            'Left2.TButton', [
                ('Button.focus', {'children': [
                    ('Button.leftarrow', None),
                    ('Button.padding', {'sticky': 'nswe', 'children': [
                        ('Button.label', {'sticky': 'nswe'}
                         )]}
                     )]}
                 )]
        )
        self.style.layout(
            'Left1.TButton', [
                ('Button.focus', {'children': [
                    ('Button.rightarrow', None),
                    ('Button.padding', {'sticky': 'nswe', 'children': [
                        ('Button.label', {'sticky': 'nswe'}
                         )]}
                     )]}
                 )]
        )
        self.shift = tk.Frame(self)
        self.shift.grid(row=22, column=1, padx=50, pady=20, sticky='we')
        self.style.configure('Left2.TButton', font=('', '15', 'bold'), width=1, arrowcolor='black')
        lbtn = ttk.Button(self.shift, style='Left2.TButton', command=self.back)
        lbtn.pack(side=LEFT)
        self.style.configure('Left1.TButton', font=('', '15', 'bold'), width=1, arrowcolor='black')
        rbtn = ttk.Button(self.shift, style='Left1.TButton', command=self.forward)
        rbtn.pack(side=RIGHT)
        cbtn = tk.Button(self, text='Delete all pairs', width=16, command=self.reset)
        cbtn.grid(row=23, column=0, ipadx=1, pady=20, columnspan=1, rowspan=1)
        dbtn = tk.Button(self, text='Delete last pair', width=16, command=self.delete_last_pairs)
        dbtn.grid(row=22, column=0, ipadx=1, pady=20, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Delete trash pairs', width=16, command=self.empty_trash)
        ebtn.grid(row=23, column=1, padx=10, pady=20, columnspan=1, rowspan=1)
        fbtn = tk.Button(self, text='Save Slam data', width=16, command=self.save_slam)
        fbtn.grid(row=24, column=1, padx=10, pady=20, columnspan=1, rowspan=1)
        fbtn = tk.Button(self, text='Save Matrix/Pairs', width=16, command=self.save_matrix)
        fbtn.grid(row=24, column=0, padx=10, pady=20, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Save Animation', width=16, command=self.gif_generator)
        ebtn.grid(row=21, column=0, padx=10, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Help', width=16, command=self.help)
        ebtn.grid(row=60, column=0, padx=10, pady=18, columnspan=2, rowspan=1)
        # ---------------------------------------------------------
        self.shift = tk.Frame(self)
        self.shift.grid(row=1, column=42, padx=20, pady=20, sticky='we')
        scale = Label(self.shift, text='Plotting scale:')
        scale.pack(side=LEFT)
        self.e1 = tk.Entry(self.shift, width=5, justify='right')
        self.e1.pack(side=LEFT)
        scale = Label(self.shift, text='\'/pixel')
        scale.pack(side=LEFT)
        label1 = Label(self, text='If you meet any bugs please contact:\nay1620@nyu.edu')
        label1.grid(row=60, column=42, padx=20, rowspan=1)
        ubtn = tk.Button(self.shift, text='Check pairs', width=10, command=self.match)
        ubtn.pack(side=LEFT)
        if os.path.exists(os.path.join(outf, 'save.json')):
            with open(os.path.join(outf, 'save.json'), 'r') as f:
                model = json.load(f)
            points3D = model['points3D']
            points2D = model['points2D']
            if len(points3D) != len(points2D):
                min_m = min(len(points3D), len(points2D))
                points3D_ = {}
                points2D = points2D[:min_m]
                n = 0
                for k, v in points3D.items():
                    if n < min_m:
                        points3D_.update({k: v})
                        n += 1
                points3D = points3D_
            T = np.array(model['Matrix'])
            num_pairs = len(points3D)
            self.plan = model['floorplan']
            self.plan=self.plan.replace('/root','/home/unav')
            try:
                self.scale = model['scale']
                self.e1.insert(END, str(self.scale))
            except:
                self.e1.insert(END, '1')
            self.match()

    def help(self):
        self.info = tk.Toplevel(self.master)
        self.info.geometry('800x650')
        self.info.title('Instruction')
        self.info.geometry("+{}+{}".format(self.positionRight - 300, self.positionDown - 200))
        # This will create a LabelFrame
        label_frame1 = LabelFrame(self.info, height=100, text='Steps')
        label_frame1.pack(expand='yes', fill='both')

        label1 = Label(label_frame1, text='1. Choose a frame in frame list.')
        label1.place(x=0, y=5)

        label2 = Label(label_frame1, text='2. Double click or press <Enter> to open frame and pick a feature point.')
        label2.place(x=0, y=35)

        label3 = Label(label_frame1,
                       text='3. Double click floor plan and choose corresponding point of previous feature point.')
        label3.place(x=0, y=65)

        label4 = Label(label_frame1, text='4. Repeat above until get good matching, save the slam data.')
        label4.place(x=0, y=95)

        label_frame1 = LabelFrame(self.info, height=400, text='Buttons')
        label_frame1.pack(expand='yes', fill='both', side='bottom')

        label_1 = LabelFrame(label_frame1, height=60, text='Save Animation')
        label_1.place(x=5, y=23)
        label1 = Label(label_1, text='Save a Gif animation of the whole mapping trajectory')
        label1.pack()

        label_2 = LabelFrame(label_frame1, height=40, text='Select Floor Plan')
        label_2.place(x=5, y=70)
        label2 = Label(label_2, text='Select a floor plan of your project.')
        label2.pack()

        label_3 = LabelFrame(label_frame1, height=40, text='Delete last pair')
        label_3.place(x=5, y=117)
        label3 = Label(label_3,
                       text='Permanently Delete last pairs that used to calculate transformation matrix.')
        label3.pack()

        label_4 = LabelFrame(label_frame1, height=60, text='Left\Right button')
        label_4.place(x=5, y=164)
        label4 = Label(label_4,
                       text='Left button moves the last current pair which used to compute matrix to trash,\nright button recovers fisrt pair in trash.')
        label4.pack()

        label_5 = LabelFrame(label_frame1, height=40, text='Delete all pairs')
        label_5.place(x=5, y=229)
        label5 = Label(label_5,
                       text='Permanently Delete all pairs in current and in trash.')
        label5.pack()

        label_6 = LabelFrame(label_frame1, height=40, text='Delete trash pairs')
        label_6.place(x=5, y=276)
        label6 = Label(label_6,
                       text='Permanently Delete all pairs in trash.')
        label6.pack()

        label_7 = LabelFrame(label_frame1, height=60, text='Save SLAM\Matrix')
        label_7.place(x=5, y=323)
        label7 = Label(label_7,
                       text='Save all current pairs, transformation matrix which will be load and reuse when reopen\nthe project. Save SLAM data which is used in colmap.')
        label7.pack()

        label_8 = LabelFrame(label_frame1, height=40, text='Animation')
        label_8.place(x=5, y=388)
        label8 = Label(label_8,
                       text='Create an animation gif of mapping.')
        label8.pack()

    def gif(self):
        c, d = int(self.e4.get()), int(self.e5.get())
        plan = Image.open(os.path.join(self.ffid[list(self.ffid.keys())[0]]['frame']))
        b, _ = plan.size
        scale = c / b
        plan = Image.open(self.plan)
        e, _ = plan.size
        gif_shrink = e / d
        if os.path.exists(os.path.join(self.outf, self.file_name)):
            with open(os.path.join(self.outf, self.file_name), 'r') as f:
                data = json.load(f)
            gif_ims = []
            T = np.array(data['T'])
            landmarks = data['landmarks']
            keyframes = data['keyframes']
            keyframes = collections.OrderedDict(natsort.natsorted(keyframes.items()))
            for i in keyframes.keys():
                frame = Image.open(os.path.join(self.src, i + '.png'))
                x_, y_ = keyframes[i]['trans'][0], keyframes[i]['trans'][1]
                ang = keyframes[i]['rot']
                kps = keyframes[i]['keypts']
                lm = [[landmarks[str(j)]['x'], landmarks[str(j)]['z']] for j in keyframes[i]['lm_ids'] if j != -1]
                draw = ImageDraw.Draw(frame)
                for p in kps:
                    xx, yy = p[0], p[1]
                    draw.rectangle([(xx - 10, yy - 10), (xx + 10, yy + 10)], fill=(0, 0, 255))
                fl = self.imf.copy()
                draw = ImageDraw.Draw(fl)

                for points in lm:
                    x, y = points
                    try:
                        tvec = self.TPS.wrap_points(np.array([[x, y]]))
                    except:
                        tvec = T @ np.array([[x], [y], [1]])
                    x, y = tvec
                    draw.rectangle([(x - 3, y - 3), (x + 3, y + 3)], fill=(0, 0, 255))
                width, height = frame.size
                newsize = (int(width * scale), int(height * scale))
                frame = frame.resize(newsize)
                width, height = frame.size
                fw, fh = fl.size
                fl.paste(frame, (fw - width, fh - height))
                for i, p in enumerate(kps):
                    x2, y2 = p[0] * scale + fw - width, p[1] * scale + fh - height
                    x3, y3 = lm[i]
                    try:
                        tvec = self.TPS.wrap_points(np.array([[x3, y3]]))
                    except:
                        tvec = T @ np.array([[x3], [y3], [1]])
                    x3, y3 = tvec
                    draw.line([(x3, y3), (x2, y2)], fill=(0, 255, 255), width=1)
                x1, y1 = x_ - 70 * np.sin(ang), y_ - 70 * np.cos(ang)
                draw.ellipse((x_ - 30, y_ - 30, x_ + 30, y_ + 30), fill=(255, 0, 255))
                draw.line([(x_, y_), (x1, y1)], fill=(255, 0, 255), width=15)
                newsize = (int(fw / gif_shrink), int(fh / gif_shrink))
                gif_ims.append(fl.resize(newsize))
            gif_ims[0].save(os.path.join(self.opt.outf, 'map_trajectory.gif'),
                            save_all=True, append_images=gif_ims[1:], optimize=False, duration=int(self.e1.get()),
                            loop=0)
            self.win.destroy()
            showinfo("Saved!", "Animation saved to:\n{}".format(os.path.join(self.opt.outf, 'map_trajectory.gif')))
        else:
            self.win.destroy()
            showinfo("Error!", "Please save SLAM data first!")

    def gif_config(self):
        self.win.destroy()
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Animation setting")
        self.win.geometry('350x150')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        self.shift = tk.Frame(self.win)
        self.shift.grid(row=0, column=0)
        self.shift1 = tk.Frame(self.shift)
        self.shift1.grid(row=0, column=0, pady=5, padx=60)
        scale = Label(self.shift1, text='Frame width in Gif:')
        scale.pack(side=LEFT)
        self.e4 = tk.Entry(self.shift1, width=4, justify='center')
        self.e4.insert(END, '450')
        self.e4.pack(side=LEFT)
        self.shift2 = tk.Frame(self.shift)
        self.shift2.grid(row=1, column=0, pady=5, padx=60)
        scale = Label(self.shift2, text='Output gif width:')
        scale.pack(side=LEFT)
        self.e5 = tk.Entry(self.shift2, width=4, justify='center')
        self.e5.insert(END, '900')
        self.e5.pack(side=LEFT)
        self.shift4 = tk.Frame(self.shift)
        self.shift4.grid(row=2, column=0, pady=5, padx=60)
        scale = Label(self.shift4, text='Frame duration:')
        scale.pack(side=LEFT)
        self.e1 = tk.Entry(self.shift4, width=2, justify='center')
        self.e1.insert(END, '30')
        self.e1.pack(side=LEFT)
        scale = Label(self.shift4, text='s')
        scale.pack(side=LEFT)
        self.shift3 = tk.Frame(self.shift)
        self.shift3.grid(row=3, column=0, pady=5)
        b1 = ttk.Button(self.win, text="Go", width=6, command=self.gif)
        b1.grid(row=3, column=0, columnspan=1, padx=30, rowspan=1, sticky='w')
        b2 = ttk.Button(self.win, text="Cancel", width=6, command=self.win.destroy)
        b2.grid(row=3, column=0, columnspan=1, padx=20, rowspan=1, sticky='e')

    def gif_generator(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('330x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Save animation to outpath?")
        l.grid(row=1, column=0, columnspan=4, rowspan=2,
               padx=20, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.gif_config)
        b1.grid(row=3, column=0, columnspan=1, padx=40, rowspan=3)

        b2 = ttk.Button(self.win, text="No", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, padx=40, rowspan=3)

    def save_data(self):
        bin_fn = self.opt.maps
        kf = {}
        lm = {}
        # scale=np.linalg.det(T[:2,:2])**(0.5)
        if opt.maps.endswith('.msg'):
            with open(bin_fn, "rb") as f:
                data = msgpack.unpackb(f.read(), use_list=False, raw=False)
            landmarks = collections.OrderedDict(natsort.natsorted(data['landmarks'].items()))
            point3d = []
            Z = []
            for t, (id, point) in enumerate(landmarks.items()):
                pos = point["pos_w"]
                point3d.append([pos[0], 0, pos[2]])
                Z.append(point["pos_w"])
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(point3d)
            cl, source1 = source.remove_radius_outlier(nb_points=16, radius=0.25)
            invalid = []
            for i, index in enumerate(landmarks.keys()):
                if i in source1:
                    # x_fp = np.array([point3d[i][0], point3d[i][2], 1]).T
                    # x_mp = T @ x_fp
                    lm.update({str(index): {'x': Z[i][0], 'y': Z[i][1], 'z': Z[i][2]}})
                else:
                    invalid.append(str(index))
            keyframes = collections.OrderedDict(natsort.natsorted(data['keyframes'].items()))
            for j, i in enumerate(list(keyframes.keys())):
                k = keyframes[i]['trans_cw']
                rot = keyframes[i]['rot_cw']
                pos = slam2world(k, rot)
                try:
                    t_mp = self.TPS.wrap_points(np.array([pos[0], pos[2]]))
                except:
                    t_fp = np.array([pos[0], pos[2], 1]).T
                    t_mp = ((T @ t_fp).T).tolist()
                kps, kes, klm = [], [], []
                for ind, k in enumerate(keyframes[i]['lm_ids']):
                    if k != -1:
                        if str(k) not in invalid:
                            kps.append(keyframes[i]['keypts'][ind]['pt'])
                            # kes.append(keyframes[i]['descs'][ind])
                            klm.append(k)
                kf.update({self.dics[j]: {'keypts': kps, 'lm_ids': klm, 'trans': t_mp,
                                          'rot': R.from_quat(rot).as_rotvec()[1] - self.rot_base}})
        else:
            loader = get_colmap_data(opt)
            images = loader.read_images_binary(loader.images_path)
            images = collections.OrderedDict(natsort.natsorted(images.items()))
            P3D = loader.read_points3d_binary(loader.point3d_path)
            point3d = []
            Z = []
            for id, point in P3D.items():
                pos = point["xyz"]
                point3d.append([pos[0], 0, pos[2]])
                Z.append(pos)
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(point3d)
            cl, source1 = source.remove_radius_outlier(nb_points=16, radius=0.25)
            invalid = []
            for i, index in enumerate(tqdm(P3D.keys(), desc='generate landmarks')):
                if i in source1:
                    lm.update({str(index): {'x': Z[i][0], 'y': Z[i][1], 'z': Z[i][2]}})
                else:
                    invalid.append(str(index))

            for j, i in enumerate(tqdm(list(images.keys()), desc='generate keyframes')):
                k = images[i]['tvec']
                rot = images[i]['qvec']
                rot = [rot[1], rot[2], rot[3], rot[0]]
                pos = slam2world(k, rot)
                try:
                    t_mp = self.TPS.wrap_points(np.array([pos[0], pos[2]]))
                except:
                    t_fp = np.array([pos[0], pos[2], 1]).T
                    t_mp = ((T @ t_fp).T).tolist()
                kps, kes, klm, ids = [], [], [], []
                for ind, k in enumerate(images[i]['point3D_ids']):
                    if k != -1:
                        if str(k) not in invalid:
                            ids.append(ind)
                            kps.append(images[i]['xys'][ind].tolist())
                            klm.append(int(k))
                kf.update({self.dics[j]: {'keypts': kps, 'lm_ids': klm, 'kp_index': ids, 'trans': t_mp,
                                          'rot': R.from_quat(rot).as_rotvec()[1] - self.rot_base}})
        try:
            coeffs = self.coeffs.tolist()
            control_pts = self.control_pts.tolist()
        except:
            coeffs = None
            control_pts = None
        return {'landmarks': lm, 'keyframes': kf, 'floorplan': self.plan, 'T': T.tolist(), 'base_rot': self.rot_base,
                'TPS_coeff': coeffs, 'control_pts': control_pts}

    def clear_plots(self):
        try:
            for widget in self.li.winfo_children():
                widget.destroy()
            self.li.destroy()
            self.sl7.destroy()
        except:
            pass
        try:
            self.l.destroy()
        except:
            pass

    def plot_pairs(self):
        self.clear_plots()
        self.li = ttk.LabelFrame(self, text='Total Pairs:%d' % num_pairs)
        self.li1 = ttk.LabelFrame(self.li, text='Index')
        self.li2 = ttk.LabelFrame(self.li, text='World\ncoordinates')
        self.li3 = ttk.LabelFrame(self.li, text='Floor Plan\ncoordinates')
        if self.linear_matching:
            self.li4 = ttk.LabelFrame(self.li, text='Error')
        self.li2x = ttk.LabelFrame(self.li2, text='x')
        self.li2y = ttk.LabelFrame(self.li2, text='y')
        self.li2z = ttk.LabelFrame(self.li2, text='z')
        self.li3x = ttk.LabelFrame(self.li3, text='X')
        self.li3y = ttk.LabelFrame(self.li3, text='Y')
        self.li2x.pack(side=LEFT)
        self.li2y.pack(side=LEFT)
        self.li2z.pack(side=LEFT)
        self.li3x.pack(side=LEFT)
        self.li3y.pack(side=LEFT)
        self.li1.pack(side=LEFT)
        self.li2.pack(side=LEFT)
        self.li3.pack(side=LEFT)
        if self.linear_matching:
            self.li4.pack(side=LEFT)
        self.sl1 = Label(self.li1, text='\n')
        self.sl1.pack()
        if self.linear_matching:
            self.sl6 = Label(self.li4, text='\n')
            self.sl6.pack()
        first_pair = len(points2D) - 20
        for i, points in enumerate(points2D):
            if (len(points) > 0) and (i >= first_pair):
                dict = list(points3D.keys())[i]
                self.sl1 = Label(self.li1, text='%-s' % (i))
                self.sl1.pack()
                self.sl2 = Label(self.li2x, text='%4.1f' % points3D[dict]['lm'][0])
                self.sl2.pack()
                self.sl2 = Label(self.li2y, text='%4.1f' % points3D[dict]['lm'][1])
                self.sl2.pack()
                self.sl2 = Label(self.li2z, text='%4.1f' % points3D[dict]['lm'][2])
                self.sl2.pack()
                self.sl3 = Label(self.li3x, text='%4d' % points[0])
                self.sl3.pack()
                self.sl3 = Label(self.li3y, text='%4d' % points[1])
                self.sl3.pack()
                if (num_pairs > 2) and self.linear_matching:
                    self.sl6 = Label(self.li4, text='%2d\'%2d\'\'' % (
                    int(self.error[i]), round((self.error[i] - int(self.error[i])) * 12)))
                    self.sl6.pack()
        if (num_pairs > 2) and self.linear_matching:
            err = np.mean(self.error)
            err_feet = int(err)
            err_inch = round((err - int(err)) * 12)
            err_cm = round(err * 30.48)
            self.sl7 = Label(self, text='Mean Error:{}\'{}\'\' ({}cm)'.format(err_feet, err_inch, err_cm))
            self.sl7.grid(row=51, column=42, rowspan=1, sticky="ew")
        self.li.grid(row=1, column=42, rowspan=50, sticky="ew")

    def local2global(self, pt):
        x = np.array([pt[0], pt[2], 1])
        global T
        return T @ x

    def matching_type(self):
        if self.v1.get() == 0:
            self.linear_matching = True
        else:
            self.linear_matching = False
            self.match()

    def match(self):
        global points3D, points2D
        if self.linear_matching:
            if len(points2D) > 2:
                p2, p3 = [], []
                for i, p in enumerate(points3D.keys()):
                    p2.append(points2D[i])
                    p3.append([points3D[p]['lm'][0], points3D[p]['lm'][2], 1])
                p2 = np.array(p2).T
                p3 = np.array(p3).T
                global T
                T = p2 @ p3.T @ np.linalg.inv(p3 @ p3.T)
                try:
                    self.scale = float(self.e1.get())
                except:
                    self.scale = 1
                self.error = np.linalg.norm(T @ p3 - p2, axis=0) * self.scale
                self.plot_pairs()
                self.rot_base = np.arctan2(T[1, 0], T[0, 0])

                x_fp = np.hstack((self.features, np.ones((self.features.shape[0], 1)))).T
                x_mp = T @ x_fp
                t_fp = np.hstack((self.kf, np.ones((self.kf.shape[0], 1)))).T
                t_mp = T @ t_fp
                self.plan=self.plan.replace('home/unav','root')
                with Image.open(self.plan) as im:
                    im=im.convert('RGB')
                    draw = ImageDraw.Draw(im)
                    for points in x_mp.T:
                        x, y = points
                        draw.rectangle([(x - 1, y - 1), (x + 1, y + 1)], fill=(0, 255, 0))
                    for points in t_mp.T:
                        x, y = points
                        draw.rectangle([(x - 2, y - 2), (x + 2, y + 2)], fill=(255, 0, 0))
                    self.imf = im.copy()
                    width, height = im.size
                    scale = 970 / width
                    newsize = (970, int(height * scale))
                im = im.resize(newsize)
                tkimage1 = ImageTk.PhotoImage(im)
                self.myvar1 = Label(self, image=tkimage1)
                self.myvar1.bind('<Double-Button-1>', self.new_window_fl)
                self.myvar1.image = tkimage1
                self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")

            else:
                self.clear_plots()
                self.l = Label(self, text="Warning: Need at least 3 pairs!")
                self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")
        else:
            if len(points2D) > 4:
                p2, p3 = [], []
                for i, p in enumerate(points3D.keys()):
                    p2.append(points2D[i])
                    p3.append([points3D[p]['lm'][0], points3D[p]['lm'][2]])
                p2 = np.array(p2)
                p3 = np.array(p3)

                self.TPS = Thin_Plate_Spline(p3, p2)
                self.coeffs = self.TPS.coeffs
                self.control_pts = p3

                try:
                    self.scale = float(self.e1.get())
                except:
                    self.scale = 1
                self.plot_pairs()
                warp_fp = self.TPS.wrap_points(self.features)
                warp_mp = self.TPS.wrap_points(self.kf)
                with Image.open(self.plan) as im:
                    im=im.convert('RGB')
                    draw = ImageDraw.Draw(im)
                    for points in warp_fp:
                        x, y = points
                        draw.rectangle([(x - 1, y - 1), (x + 1, y + 1)], fill=(0, 255, 0))
                    for points in warp_mp:
                        x, y = points
                        draw.rectangle([(x - 2, y - 2), (x + 2, y + 2)], fill=(255, 0, 0))
                    self.imf = im.copy()
                    width, height = im.size
                    scale = 970 / width
                    newsize = (970, int(height * scale))
                im = im.resize(newsize)
                tkimage1 = ImageTk.PhotoImage(im)
                self.myvar1 = Label(self, image=tkimage1)
                self.myvar1.bind('<Double-Button-1>', self.new_window_fl)
                self.myvar1.image = tkimage1
                self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")

            else:
                self.clear_plots()
                self.l = Label(self, text="Warning: Need at least 5 pairs!")
                self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")

    def empty(self):
        if len(self.points3D) > 0:
            self.points3D = {}
            self.points2D = []
            self.win.destroy()
            showinfo("Empty!", "All pairs in trash have been removed")
        else:
            self.win.destroy()
            showinfo("Error!", "No pairs in trash!")

    def empty_trash(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('320x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Do you confirm empty trash?")
        l.grid(row=1, column=0, columnspan=2, rowspan=2,
               padx=40, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.empty)
        b1.grid(row=3, column=0, columnspan=1, rowspan=3)

        b2 = ttk.Button(self.win, text="Cancel", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, rowspan=3)

    def save_map(self):
        global points3D, points2D, T
        if len(points3D) > 2:
            if self.save:
                data = self.save_data()
                with open(os.path.join(self.outf, self.file_name), 'w') as f:
                    json.dump(data, f)
                self.save_topomap = True
                self.win.destroy()
                showinfo("Saved!", "slam data saved successfully")
            else:
                self.win.destroy()
                showinfo("Error!", "Please save matrix/pairs first!")
        else:
            self.win.destroy()
            showinfo("Error!", "No map available")

    def save_mat(self):
        global points3D, points2D, T
        if len(points3D) > 2:
            with open(os.path.join(self.outf, 'save.json'), 'w') as f:
                json.dump({'Matrix': T.tolist(), 'points3D': points3D, 'points2D': points2D, 'floorplan': self.plan,
                           'scale': self.scale}, f)
            self.save = True
            self.win.destroy()
            showinfo("Saved!", "Matrix and pairs saved successfully")
        else:
            self.win.destroy()
            showinfo("Error!", "No matrix available")

    def save_matrix(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('380x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Save transformation matrix and pairs?")
        l.grid(row=1, column=0, columnspan=4, rowspan=2,
               padx=20, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.save_mat)
        b1.grid(row=3, column=0, columnspan=1, padx=40, rowspan=3)

        b2 = ttk.Button(self.win, text="No", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, padx=40, rowspan=3)

    def save_slam(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('330x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Save slam data?")
        l.grid(row=1, column=0, columnspan=4, rowspan=2,
               padx=20, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.save_map)
        b1.grid(row=3, column=0, columnspan=1, padx=40, rowspan=3)

        b2 = ttk.Button(self.win, text="No", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, padx=40, rowspan=3)

    def delete_last_pair(self):
        global points3D, points2D
        if (len(points3D) > 0) and (len(points2D) == len(points3D)):
            points3D.pop(list(points3D.keys())[-1])
            points2D.pop()
            if len(self.points3D) > 0:
                for i in list(self.points3D.keys())[::-1]:
                    self.points3D.update({str(int(i) - 1).zfill(3): self.points3D[i]})
                self.points3D.pop(list(self.points3D.keys())[0])
            global num_pairs
            num_pairs -= 1
            self.save = False
            self.win.destroy()
            showinfo("Deleted!", "Last current Pairs have been removed")
            if len(points3D) > 2:
                self.match()
            else:
                self.plot_pairs()
                im = Image.open(self.plan)
                im=im.convert('RGB')
                width, height = im.size
                scale = 970 / width
                newsize = (970, int(height * scale))
                im = im.resize(newsize)
                tkimage1 = ImageTk.PhotoImage(im)
                self.myvar1 = Label(self, image=tkimage1)
                self.myvar1.bind('<Double-Button-1>', self.new_window_fl)
                self.myvar1.image = tkimage1
                self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")
        else:
            self.win.destroy()
            showinfo("Error!", "No pairs left or number of pairs isn't matched!")

    def delete_last_pairs(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('380x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Do you confirm delete last pairs?")
        l.grid(row=1, column=0, columnspan=2, rowspan=2,
               padx=40, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.delete_last_pair)
        b1.grid(row=3, column=0, columnspan=1, rowspan=3)

        b2 = ttk.Button(self.win, text="Cancel", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, rowspan=3)

    def delete_points(self):
        global points3D, points2D, num_pairs, FL, Frame
        FL = False
        Frame = True
        num_pairs = 0
        points3D, points2D, self.points3D, self.points2D = {}, [], {}, []
        self.save = False
        self.win.destroy()
        showinfo("Deleted!", "All Pairs have been deleted")
        if len(points3D) > 2:
            self.match()
        else:
            self.plot_pairs()
            im = Image.open(self.plan)
            im=im.convert('RGB')
            width, height = im.size
            scale = 970 / width
            newsize = (970, int(height * scale))
            im = im.resize(newsize)
            tkimage1 = ImageTk.PhotoImage(im)
            self.myvar1 = Label(self, image=tkimage1)
            self.myvar1.bind('<Double-Button-1>', self.new_window_fl)
            self.myvar1.image = tkimage1
            self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")

    def reset(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('380x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Do you confirm delete all pairs?")
        l.grid(row=1, column=0, columnspan=2, rowspan=2,
               padx=40, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.delete_points)
        b1.grid(row=3, column=0, columnspan=1, rowspan=3)

        b2 = ttk.Button(self.win, text="Cancel", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, rowspan=3)

    def back(self):
        if len(points3D) > len(points2D):
            self.clear_plots()
            self.l = Label(self, text="Warning: Please Pick floor plan points of\n new frame points before back!")
            self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")
        elif len(points3D) == 0:
            self.clear_plots()
            self.l = Label(self, text="Warning: No current pairs exist!")
            self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")
        else:
            self.points3D.update({list(points3D.keys())[-1]: points3D[list(points3D.keys())[-1]]})
            self.points2D.append(points2D[-1])
            points3D.pop(list(points3D.keys())[-1])
            points2D.pop()
            global num_pairs
            num_pairs -= 1
            if len(points3D) > 2:
                self.match()
            else:
                self.plot_pairs()
                im = Image.open(self.plan)
                im=im.convert('RGB')
                width, height = im.size
                scale = 970 / width
                newsize = (970, int(height * scale))
                im = im.resize(newsize)
                tkimage1 = ImageTk.PhotoImage(im)
                self.myvar1 = Label(self, image=tkimage1)
                self.myvar1.bind('<Double-Button-1>', self.new_window_fl)
                self.myvar1.image = tkimage1
                self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")

    def forward(self):
        if len(points3D) > len(points2D):
            self.clear_plots()
            self.l = Label(self, text="Warning: Please Pick floor plan points of\n new frame points before forward!")
            self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")
        elif len(self.points3D) == 0:
            self.clear_plots()
            self.l = Label(self, text="Warning: No thrown pairs exist!")
            self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")
        else:
            points3D.update({list(self.points3D.keys())[-1]: self.points3D[list(self.points3D.keys())[-1]]})
            points2D.append(self.points2D[-1])
            self.points3D.pop(list(self.points3D.keys())[-1])
            self.points2D.pop()
            global num_pairs
            num_pairs += 1
            if len(points3D) > 2:
                self.match()
            else:
                self.plot_pairs()
                im = Image.open(self.plan)
                im=im.convert('RGB')
                width, height = im.size
                scale = 970 / width
                newsize = (970, int(height * scale))
                im = im.resize(newsize)
                tkimage1 = ImageTk.PhotoImage(im)
                self.myvar1 = Label(self, image=tkimage1)
                self.myvar1.bind('<Double-Button-1>', self.new_window_fl)
                self.myvar1.image = tkimage1
                self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")

    def plot_matches(self):
        if len(points3D) == len(points2D):
            self.plot_pairs()
        else:
            self.clear_plots()
            self.l = Label(self, text="Warning: Please Pick floor plan points of\n new frame points!")
            self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")

    def show_frame(self, action):
        if Frame:
            self.value = self.lb.get(self.lb.curselection())
            if action == 'up':
                i = self.dics.index(self.value)
                if i > 0:
                    self.value = self.dics[i - 1]
            if action == 'down':
                i = self.dics.index(self.value)
                if i < (len(self.dics) - 1):
                    self.value = self.dics[i + 1]
            with Image.open(os.path.join(self.src, self.value + '.png')) as im:
                im=im.convert('RGB')
                draw = ImageDraw.Draw(im)
                chose = []
                global points3D
                for j in points3D.keys():
                    if points3D[j]['id'] == self.value:
                        x, y = points3D[j]['frame']
                        draw.rectangle([(x - 20, y - 20), (x + 20, y + 20)], fill=(255, 255, 0))
                        chose.append([x, y])
                for points in self.ffid[self.value]['gp']:
                    x, y = points
                    if [x, y] not in chose:
                        draw.rectangle([(x - 10, y - 10), (x + 10, y + 10)], fill=(0, 255, 0))
            self.fm = im.copy()
            framew, frameh = im.size
            scale = 600 / framew
            newsize = (600, int(frameh * scale))
            im = im.resize(newsize)
            tkimage = ImageTk.PhotoImage(im)
            self.myvar = Label(self, image=tkimage)
            self.myvar.bind('<Double-Button-1>', self.new_window_feature)
            self.myvar.image = tkimage
            self.myvar.grid(row=1, column=4, columnspan=1, rowspan=10, sticky="snew")
            if (num_pairs > 2) and self.linear_matching:
                ind = self.dics.index(self.value)
                t_fp = np.array([self.kf[ind, 0], self.kf[ind, 1], 1]).T
                ang = self.ffid[self.value]['ang'] - self.rot_base
                t_mp = T @ t_fp
                lm = np.array([self.local2global(i) for i in self.ffid[self.value]['lm']])
                x_fp = np.hstack((self.features, np.ones((self.features.shape[0], 1)))).T
                x_mp = T @ x_fp
                index = []
                lm1 = []
                for j, i in enumerate(lm):
                    if i in x_mp.T:
                        lm1.append(i)
                        index.append(j)
                lm = np.array(lm1)
                im = self.imf.copy()
                draw = ImageDraw.Draw(im)
                for points in lm:
                    x, y = points
                    draw.rectangle([(x - 3, y - 3), (x + 3, y + 3)], fill=(0, 0, 255))
                x, y = t_mp.T
                x1, y1 = x - 70 * np.sin(ang), y - 70 * np.cos(ang)
                draw.ellipse((x - 30, y - 30, x + 30, y + 30), fill=(255, 0, 255))
                draw.line([(x, y), (x1, y1)], fill=(255, 0, 255), width=15)
                self.imfc = im.copy()
                width, height = im.size
                scale = 970 / width
                newsize = (970, int(height * scale))
                im = im.resize(newsize)
                tkimage1 = ImageTk.PhotoImage(im)
                self.myvar1 = Label(self, image=tkimage1)
                self.myvar1.bind('<Double-Button-1>', self.new_window_fl)
                self.myvar1.image = tkimage1
                self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")
            elif (num_pairs > 4):
                ind = self.dics.index(self.value)
                pt = self.kf[ind]
                warp_mp = self.TPS.wrap_points(np.array([[pt[0], pt[1]]]))
                ang = self.ffid[self.value]['ang']
                pt1 = np.array([[pt[0] + self.rot_scale * np.cos(ang), pt[1] + self.rot_scale * np.sin(ang)]])
                warp_mp1 = self.TPS.wrap_points(pt1)
                ang = -np.arctan2(warp_mp1[0, 1] - warp_mp[0, 1], warp_mp1[0, 0] - warp_mp[0, 0])
                im = self.imf.copy()
                draw = ImageDraw.Draw(im)
                # lm =self.TPS.wrap_points(np.array(self.ffid[self.value]['lm'])[:,:2])
                # warp_fp=self.TPS.wrap_points(self.features)
                # index = []
                # lm1 = []
                # for j, i in enumerate(lm):
                #     if i in warp_fp:
                #         lm1.append(i)
                #         index.append(j)
                # lm = np.array(lm1)
                # for points in lm:
                #     x, y = points
                #     draw.rectangle([(x - 3, y - 3), (x + 3, y + 3)], fill=(0, 0, 255))
                x, y = warp_mp[0]
                x1, y1 = x - 70 * np.sin(ang), y - 70 * np.cos(ang)
                draw.ellipse((x - 30, y - 30, x + 30, y + 30), fill=(255, 0, 255))
                draw.line([(x, y), (x1, y1)], fill=(255, 0, 255), width=15)
                self.imfc = im.copy()
                width, height = im.size
                scale = 970 / width
                newsize = (970, int(height * scale))
                im = im.resize(newsize)
                tkimage1 = ImageTk.PhotoImage(im)
                self.myvar1 = Label(self, image=tkimage1)
                self.myvar1.bind('<Double-Button-1>', self.new_window_fl)
                self.myvar1.image = tkimage1
                self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")
        else:
            self.clear_plots()
            self.l = Label(self, text="Warning: Please Pick floor plan points of\n new frame 3D points!")
            self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")

    def FloorPlan_select(self):
        fl_selected = filedialog.askopenfilename(initialdir=opt.plan, title='Select Floor Plan')
        self.plan = fl_selected
        im = Image.open(fl_selected)
        im=im.convert('RGB')
        width, height = im.size
        scale = 970 / width
        newsize = (970, int(height * scale))
        im = im.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(im)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.bind('<Double-Button-1>', self.new_window_fl)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")

    def new_window_feature(self, w):
        if Frame:
            global num_pairs
            self.value = self.lb.get(self.lb.curselection())
            self.newWindow = tk.Toplevel(self.master)
            self.app1 = Features_window(self.newWindow, parent=self, data=self.ffid[self.value], name=self.value)
        else:
            self.clear_plots()
            self.l = Label(self, text="Warning: Please Pick floor plan points of\n new frame 3D points!")
            self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")

    def new_window_fl(self, w):
        if FL:
            self.value = self.lb.get(self.lb.curselection())
            self.newWindow = tk.Toplevel(self.master)
            self.app2 = FloorPlan_window(self.newWindow, parent=self, data=self.ffid[self.value], name=self.plan)
        else:
            self.clear_plots()
            self.l = Label(self, text="Warning: Please Pick a frame feature point \nFirst!")
            self.l.grid(row=2, column=41, columnspan=20, rowspan=40, sticky="ew")


def main(opt):
    ffid, features, kf = load_data(opt)
    style = Style(theme='superhero')
    root = style.master
    Main_window(root, ffid, opt.src_dir, features, kf, opt.outf, opt, opt.file_name)
    root.mainloop()


if __name__ == '__main__':
    opt = options()
    main(opt)
