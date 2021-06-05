# Simple enough, just import everything from tkinter.
from tkinter import *
from tkinter import filedialog

# download and install pillow:
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow
from PIL import Image, ImageTk

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

import util as util
import threading

from gtts import gTTS
import pygame
import io


if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# This is needed to display the images.
# %matplotlib inline

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

from itertools import count, cycle

class ImageLabel(Label):
    """
    A Label that displays images, and plays them if they are gifs

    :im: A PIL Image instance or a string filename
    """
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        frames = []

        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)


# Only GrapheFrame class is for graphe
# the rest of code is for app
import math
class GrapheFrame(Frame):

    def __init__(self, master=None, spatials=None, names=None, boxes=None):
        Frame.__init__(self, master)
        self.master = master

        self.spatials = spatials
        self.names = names
        self.boxes = boxes

        self.scale = 1.0

        self.initUI()

    def move_from(self, event):
        # ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        # ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.delta <= -1:
            self.scale = 1.2
        elif event.delta >= 1:
            self.scale = 0.8
        print(str(event.delta) + ', ' + str(self.scale))
        # Rescale all canvas objects
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.scale('all', x, y, self.scale, self.scale)
        # self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def initUI(self):

        self.master.title("Graph")
        self.pack(fill=BOTH, expand=1)

        def onZoomOut():
            self.canvas.scale('all', 0, 0, 1.2, 1.2)

        def onZoomIn():
            self.canvas.scale('all', 0, 0, 0.8, 0.8)

        self.btnZoomIn = Button(self.master, text="Zoom Out", width=10, command=onZoomIn)
        self.btnZoomIn.pack()
        self.btnZoomIn.place(x=0, y=0)
        self.btnZoomOut = Button(self.master, text="Zoom In", width=10, command=onZoomOut)
        self.btnZoomOut.pack()
        self.btnZoomOut.place(x=80, y=0)

        self.canvas = Canvas(self)
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>', self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.wheel)  # only with Linux, wheel scroll up

        for i in range(len(self.boxes)):
            x = self.boxes[i][5]
            y = self.boxes[i][4]
            self.canvas.create_text(x, y, anchor=W, font="Purisa", text=self.names[i])

            for j in range(8):
                neighbor = self.spatials[i][j]
                if neighbor != -1:
                    x1 = self.boxes[neighbor][5]
                    y1 = self.boxes[neighbor][4]
                    self.canvas.create_line(x, y, x1, y1)

                    dx = x1 - x
                    dy = y1 - y
                    ldx = math.sqrt(dx*dx + dy*dy)
                    alpha = 10
                    dx = alpha * dx / ldx
                    dy = alpha * dy / ldx
                    if j == 0:  #left
                        txt = 'right'
                    elif j == 1:
                        txt = 'left'
                    elif j == 2:
                        txt = 'below'
                    elif j == 3:
                        txt = 'above'
                    elif j == 4:
                        txt = 'bottom right'
                    elif j == 5:
                        txt = 'bottom left'
                    elif j == 6:
                        txt = 'top right'
                    elif j == 7:
                        txt = 'top left'

                    self.canvas.create_text(x + dx, y + dy, anchor=W, font=("Purisa", 8), text=txt)

        self.canvas.pack(fill=BOTH, expand=1)

# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

        self.imgLabel = None

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a menu instance
        self.menu = Menu(self.master)
        self.master.config(menu=self.menu)

        # create the file object)
        self.file = Menu(self.menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        self.file.add_command(label="Open", command=self.openFile)
        self.file.add_command(label='say description', command=self.say_description)
        self.file.add_command(label='show graph', command=self.show_graph)
        self.file.add_command(label="Exit", command=self.client_exit)

        self.file.entryconfig(2, state=DISABLED)
        self.file.entryconfig(3, state=DISABLED)

        # added "file" to our menu
        self.menu.add_cascade(label="File", menu=self.file)

        # create the file object)
        # edit = Menu(menu)
        # #
        # # # adds a command to the menu option, calling it exit, and the
        # # # command it runs on event is client_exit
        # # edit.add_command(label="Show Img", command=self.showImg)
        # edit.add_command(label="Show Text", command=self.showText)
        #
        # # added "file" to our menu
        # menu.add_cascade(label="Edit", menu=edit)


        t = Text(self.master, width=50, height=1)
        t.pack()
        t.place(x=145, y=5)

        # l1 = Label(self.master, text="Where is the ")
        # l2 = Label(self.master, text=" in the image?")
        # l1.pack()
        # l1.place(x=85, y=5)
        # l2.pack()
        # l2.place(x=345, y=5)

        def callback():

            print(t.get('1.0', 'end'))
            ask = t.get('1.0', 'end')
            ask = ask.lower().rstrip()
            # i = self.names[name]
            #
            # bExist = False
            # for i, n in self.names.items():
            #     if n == name:
            #         bExist = True
            #         break
            #
            # if bExist == True:
            #     neighbors = []
            #     positions = []
            #     for p in range(4):
            #         if self.spatials[i][p] != -1:
            #             neighbors.append(self.spatials[i][p])
            #             positions.append(p)
            #
            #
            #     import random
            #     pos = -1
            #     n = len(neighbors)
            #     if n > 0:
            #         m = random.randint(0, n - 1)
            #         neighbor = neighbors[m]
            #         pos = positions[m]
            #
            #     if pos == 0:    # left
            #         question = self.names[i] + ' is on the right of ' + self.names[neighbor]
            #     elif pos == 1:    # right
            #         question = self.names[i] + ' is on the left of ' + self.names[neighbor]
            #     elif pos == 2:    # above
            #         question = 'There is ' + self.names[i] + ' below ' + self.names[neighbor]
            #     elif pos == 3:    # below
            #         question = 'There is ' + self.names[i] + ' above ' + self.names[neighbor]
            #     else:   # alone
            #         question = self.names[i] + 'is alone'
            # else:
            #     question = 'name doesn\'t exist in the image'

            answer = util.process_ask(ask, self.spatials, self.names, self.classes, category_index)
            print(answer)
            self.text_to_speech(answer)

            self.file.entryconfig(1, state=NORMAL)
            self.btnAsk.config(state="normal")
            self.file.entryconfig(2, state=NORMAL)
            self.file.entryconfig(3, state=NORMAL)

        def onAsk():
            self.file.entryconfig(1, state=DISABLED)
            self.btnAsk.config(state="disabled")
            self.file.entryconfig(2, state=DISABLED)
            self.file.entryconfig(3, state=DISABLED)

            t = threading.Thread(target=callback)
            t.start()

        self.btnAsk = Button(self.master, text="Ask", width=10, command=onAsk)
        self.btnAsk.pack()
        self.btnAsk.place(x=0, y=0)
        self.btnAsk.config(state="disabled")

    def text_to_speech(self, text):
        tts = gTTS(text)
        pygame.mixer.init()
        pygame.init()  # this is needed for pygame.event.* and needs to be called after mixer.init() otherwise no sound is played
        with io.BytesIO() as f:  # use a memory stream
            tts.write_to_fp(f)
            f.seek(0)
            pygame.mixer.music.load(f)
            pygame.mixer.music.set_endevent(pygame.USEREVENT)
            pygame.event.set_allowed(pygame.USEREVENT)
            pygame.mixer.music.play()
            pygame.event.wait()
            f.close()

    def say_description(self):
        def callback():
            self.text_to_speech(self.description)
            self.file.entryconfig(1, state=NORMAL)
            self.btnAsk.config(state="normal")
            self.file.entryconfig(2, state=NORMAL)
            self.file.entryconfig(3, state=NORMAL)

        t = threading.Thread(target=callback)
        t.start()
        self.file.entryconfig(1, state=DISABLED)
        self.btnAsk.config(state="disabled")
        self.file.entryconfig(2, state=DISABLED)
        self.file.entryconfig(3, state=DISABLED)

    def show_graph(self):
        newwin = Toplevel(self.master)
        graphFrame = GrapheFrame(newwin, self.spatials, self.names, self.boxes)
        newwin.geometry(str(self.im_width) + 'x' + str(self.im_height))

        graphFrame.pack()

    def openFile(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                   filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))

        if filename == '':
            return

        image = Image.open(filename)
        BASE_IMAGE_SIZE = 800
        wpercent = (BASE_IMAGE_SIZE / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((BASE_IMAGE_SIZE, hsize), Image.ANTIALIAS)

        def callback():
            self.file.entryconfig(1, state=DISABLED)
            self.btnAsk.config(state="disabled")
            self.file.entryconfig(2, state=DISABLED)
            self.file.entryconfig(3, state=DISABLED)

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
            # Visualization of the results of a detection.
            _, self.boxes, self.names, self.classes, size = util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)

            # plt.figure(figsize=IMAGE_SIZE)
            # plt.imshow(image_np)
            # Image.fromarray(image_np).save(image_path + ".out.jpg")
            #
            # boxes, classes, scores, size = util.get_detected_objects(
            #     image_np,
            #     output_dict['detection_boxes'],
            #     output_dict['detection_classes'],
            #     output_dict['detection_scores'],
            #     category_index)

            # util.spatioa_logic is a function to calculate spatial logci.
            self.spatials = util.spatial_logic(self.boxes, size)

            im_width, im_height = image.size
            self.description = util.generate_description(self.boxes, self.names, self.classes, im_width, im_height)
            print(self.description)

            render = ImageTk.PhotoImage(Image.fromarray(image_np))
            self.imgLabel = Label(self, image=render)
            self.imgLabel.image = render
            self.imgLabel.place(x=0, y=25)

            image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
            self.im_width, self.im_height = image_pil.size
            self.master.geometry(str(self.im_width) + "x" + str(self.im_height))

            self.imgLoading.destroy()
            self.loaded = True

            self.file.entryconfig(1, state=NORMAL)
            self.btnAsk.config(state="normal")
            self.file.entryconfig(2, state=NORMAL)
            self.file.entryconfig(3, state=NORMAL)

        self.imgLoading = ImageLabel(self.master)
        self.imgLoading.load("loading.gif")
        self.imgLoading.place(x=0, y=25)
        self.master.geometry("800x600")

        if self.imgLabel != None:
            self.imgLabel.destroy()

        t = threading.Thread(target=callback)
        t.start()

    def showImg(self):
        load = Image.open("chat.png")
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

    def showText(self):
        text = Label(self, text="Hey there good lookin!")
        text.pack()

    def client_exit(self):
        exit()


# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = Tk()

root.geometry("800x600")

# creation of an instance
app = Window(root)
root.resizable(width=FALSE, height=FALSE)

# mainloop
root.mainloop()