import matplotlib.pyplot as plt
import numpy as np
import os
import pyglet
from queue import Queue
import subprocess as sp
import threading


class Q_Map_Renderer:
    def __init__(self, path, viewer=False):

        def renderer(rendering_queue, viewer):
            color_map = plt.get_cmap('inferno')
            n_recordings = 0

            while True:
                data, name = rendering_queue.get()
                print('[renderer] preparing next video... {} in the queue'.format(rendering_queue.qsize()))
                self.process_frames(data, name, color_map, viewer)
                n_recordings += 1

        self.videos_path = path + '/videos/'
        self.visits_path = path + '/visits/'
        self.coords_path = path + '/coords/'
        for _path in [self.videos_path, self.visits_path, self.coords_path]:
            if not os.path.exists(_path):
                os.makedirs(_path)

        self.rendering_queue = Queue()
        if viewer:
            self.viewer = SimpleImageViewer()
        else:
            self.viewer = None
        self.renderer_thread = threading.Thread(target=renderer, args=(self.rendering_queue, self.viewer))
        self.renderer_thread.daemon = True
        self.renderer_thread.start()

        self.buffer = []
        self.rc_counts = np.zeros((1, 1))

    def add(self, ob, coords_shape, q_values, action, action_type, n_acts, candidates, biased_candidates, goal):
        self.buffer.append((ob, coords_shape, q_values, action, action_type, n_acts, candidates, biased_candidates, goal))
        # even if we do not render the current episode we still keep track of the visitation counts
        _, _, _, (full_r, full_c) = ob
        n_rows, n_cols = prev_n_rows, prev_n_cols = self.rc_counts.shape
        if full_r+1 > n_rows: n_rows = full_r+1
        if full_c+1 > n_cols: n_cols = full_c+1
        if (n_rows, n_cols) != self.rc_counts.shape:
            self.new_rc_counts = np.zeros((n_rows, n_cols), int)
            self.new_rc_counts[:prev_n_rows, :prev_n_cols] = self.rc_counts
            self.rc_counts = self.new_rc_counts
        self.rc_counts[full_r, full_c] += 1

    def render(self, name):
        self.rendering_queue.put((self.buffer, name))
        self.buffer = []

    def reset(self):
        self.buffer = []

    def process_frames(self, data, name, color_map, viewer):
        coordinates = []

        for i, (ob, coords_shape, q_values, action, action_type, n_acts, candidates, biased_candidates, goal) in enumerate(data):
            frames, (r, c, w), screen, (full_r, full_c) = ob
            frames = np.array(frames)
            screen = np.array(screen)
            coordinates.append([full_r, full_c])

            row, col = r, c - w
            if goal is not None:
                goal_row, goal_col = goal[0], goal[1] - w
            img_height, img_width, _ = screen.shape
            margin = 2
            message_height = 8

            canvas = np.full((message_height+img_height+3*margin, 3*img_width + 4*margin, 3), 255, np.uint8)
            imgs = [
                canvas[margin:margin+message_height, (canvas.shape[1]-46)//2:(canvas.shape[1]-46)//2+46],
                canvas[margin+message_height+margin:margin+message_height+margin+img_height, margin:margin+img_width],
                canvas[margin+message_height+margin:margin+message_height+margin+img_height, margin+img_width+margin:margin+img_width+margin+img_width],
                canvas[margin+message_height+margin:margin+message_height+margin+img_height, margin+img_width+margin+img_width+margin:margin+img_width+margin+img_width+margin+img_width]
            ]

            if i == 0:
                q_map_file_name = self.videos_path + name + '.mp4'
                resolution = '{}x{}'.format(canvas.shape[1], canvas.shape[0])
                fps = 60
                # command = ['ffmpeg', '-y', '-f', 'rawvideo', '-s', resolution, '-pixel_format', 'rgb24', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0', '-pix_fmt', 'yuv420p', q_map_file_name]
                command = ['ffmpeg', '-y', '-f', 'rawvideo', '-s', resolution, '-pixel_format', 'rgb24', '-r', str(fps), '-i', '-', '-codec', 'png', q_map_file_name]
                fnull = open(os.devnull, 'w')
                pipe = sp.Popen(command, stdin=sp.PIPE, stdout=fnull, stderr=fnull)

            # draw the action type
            if action_type == 'random':
                message = [
                    [1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1],
                    [1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,1,1],
                    [1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0,1],
                    [1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1],
                    [1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1],
                    [1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1],
                    [1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ]
            elif action_type == 'dqn':
                message = [
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ]
            elif action_type == 'qmap':
                message = [
                    [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ]
            elif action_type == 'dqn/qmap':
                message = [
                    [0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                ]
            imgs[0][:] = 255 * (1 - np.stack([message]*3, 2))

            # draw the input frames + candidates + goal if exist + coordinates
            screen_ob_ratio = img_height // frames.shape[0]
            frames = frames[:, :, -3:]
            img = np.repeat(np.repeat(frames, screen_ob_ratio, 0), screen_ob_ratio, 1)
            # coords
            screen_coords_ratio = img_height // coords_shape[0]
            img[row*screen_coords_ratio:(row+1)*screen_coords_ratio, col*screen_coords_ratio:(col+1)*screen_coords_ratio] = [255,223,0]

            # goal
            if goal is not None:
                if action_type == 'dqn/qmap': goal_color = [0, 150, 0]
                else: goal_color = [220, 0, 0]
                img[goal_row*screen_coords_ratio:(goal_row+1)*screen_coords_ratio, goal_col*screen_coords_ratio:(goal_col+1)*screen_coords_ratio] = goal_color
            # candidates
            if len(candidates) > 0:
                for cr, cc in candidates:
                    img[cr*screen_coords_ratio:(cr+1)*screen_coords_ratio, cc*screen_coords_ratio:(cc+1)*screen_coords_ratio] = [220, 0, 0]
            # biased candidates
            if len(biased_candidates) > 0:
                for cr, cc in biased_candidates:
                    img[cr*screen_coords_ratio:(cr+1)*screen_coords_ratio, cc*screen_coords_ratio:(cc+1)*screen_coords_ratio] = [0, 150, 0]
            imgs[1][:] = img

            # draw the maximum q_values
            if q_values is not None:
                img = q_values.max(2)
                # color map automatically clips to [0., 1.] if floats otherwise [0, 255]
                # and returns values in [0., 1.]
                img = (color_map(img)[:, :, :3] * 255).astype(np.uint8)
                img = np.repeat(np.repeat(img, screen_coords_ratio, 0), screen_coords_ratio, 1)
                imgs[2][:] = img

            # draw the screen
            imgs[3][:] = screen

            if viewer is not None:
                viewer.imshow(canvas)

            pipe.stdin.write(canvas.tostring())

        visits_file_name = self.visits_path + name + '.npy'
        coordinates_file_name = self.coords_path + name + '.npy'

        pipe.stdin.close()
        pipe.wait()
        del pipe
        print('created video', q_map_file_name)

        np.save(visits_file_name, self.rc_counts)
        print('created visits numpy file', visits_file_name)

        np.save(coordinates_file_name, np.array(coordinates, dtype=np.int32))
        print('created coordinates numpy file', coordinates_file_name)


class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        height, width = arr.shape[:2]
        desired_height, desired_width = 1000, 1000
        up_scale = min(desired_height // height, desired_width // width)
        arr = np.repeat(np.repeat(arr, up_scale, 0), up_scale, 1)

        if self.window is None:
            height, width, _channels = arr.shape
            self.window = pyglet.window.Window(width=1*width, height=1*height, display=self.display, vsync=False, resizable=False)
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1]*-3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0, width=self.window.width, height=self.window.height)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
