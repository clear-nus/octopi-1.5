import tkinter as tk
from tkinter import scrolledtext, Canvas, Frame, StringVar
import requests
import ast
import io
import threading

import cv2
import numpy as np
from threading import Thread, Lock
from tqdm import tqdm
import os, shutil
import random, glob
import time
import subprocess

CAMERA_CAPTURE_FREQUENCY = 60 # tested, do not increase!


# OPTICAL FLOW CONFIG
COMPUTE_OF = False # compute optical flow
WIDTH = 720
HEIGHT = 960
RESIZED_FOR_OF = True # set true for optical flow

###################
# localised TMI GUI instance that also contains the primary Gelsight capture functionality
###################

def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))
    # keep the ratio the same as the original image size
    img = img[border_size_x+2:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    # final resize for 3d
    img = cv2.resize(img, (imgw, imgh))
    return img

def downscale_image(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Get original dimensions
    height, width = image.shape[:2]

    # Compute new dimensions (20% of original size)
    new_width = int(width * 0.2)
    new_height = int(height * 0.2)
    new_dim = (new_width, new_height)

    # Resize the image
    downscaled_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    # Save the new image
    cv2.imwrite(output_path, downscaled_image)

def search_for_devices(total_ids = 10):
    '''
    Found gelsights for the sensor.
    '''

    found_ids = []

    print("Looking for gelsight devices")
    for _id in range(0, total_ids, 2):
        try:
            cap = cv2.VideoCapture(_id)
            ret, _ = cap.read()
            if ret:
                found_ids.append(_id)
                cap.release()
            else:
                break
        except:
            pass

    print('Found {} gelsight sensors.'.format(len(found_ids)))

    return found_ids    

class WebcamVideoStream :
    '''
    Thread based visualization
    '''
    def __init__(self, src, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        assert self.grabbed != False, "Camera with src={} is not found".format(src)

        self.started = False
        self.read_lock = Lock()

        self.connection_lost = False
        self.src = src

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            try:
                (grabbed, frame) = self.stream.read()
                self.read_lock.acquire()
                self.grabbed, self.frame = grabbed, frame

                if self.frame is None:
                    self.connection_lost = True
                    print("lost connection with camera, src={}".format(self.src))
                    print("Shutting down")
                    exit()
                self.read_lock.release()
            except:
                continue

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

class Octopi:
    def __init__(self, root, src, width=240, height=320):

        ######## TMI GUI
        self.root = root
        self.root.title("Octopi")
        self.root.geometry("550x650")

        self.rank_criteria = ""
        self.listening_for_objects = False
        self.all_items = ""

        # Scrollable message area
        self.canvas = Canvas(self.root, width=400, height=500)
        self.canvas.pack(pady=10, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.message_frame = Frame(self.canvas)
        self.message_frame.bind("<Configure>", self.on_frame_configure)

        # Attach the frame to the canvas
        self.canvas.create_window((0, 0), window=self.message_frame, anchor="nw")

        self.status_message = StringVar()
        self.status = tk.Label(self.root, textvariable=self.status_message, wraplength=300, justify="left",
                                               anchor="w", padx=10, pady=5, font=("Helvetica", 10))
        self.status.pack(pady=0)
        self.status_message.set("")

        # Entry box for typing messages
        self.entry_box = tk.Entry(self.root, width=70, font=('Helvetica 24'))
        self.entry_box.pack(pady=5)

        # Button frame at the bottom
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10, side=tk.BOTTOM)



        # Buttons
        self.touch_items_btn = tk.Button(button_frame, text="Touch Item", width=10, command=self.touch_new_item)
        self.touch_items_btn.grid(row=0, column=0, padx=5)

        self.touch_items_btn = tk.Button(button_frame, text="Take Picture", width=10, command=self.take_pic)
        self.touch_items_btn.grid(row=0, column=1, padx=5)

        self.touch_this_btn = tk.Button(button_frame, text="Touch Again", width=10, command=self.touch_this)
        self.touch_this_btn.grid(row=0, column=2, padx=5)

        self.send_btn = tk.Button(button_frame, text="Send", width=10, command=self.send_message)
        self.send_btn.grid(row=0, column=3, padx=5)

        self.reset_btn = tk.Button(button_frame, text="Reset", width=10, bg='red', fg='white', command=self.reset_chat)
        self.reset_btn.grid(row=0, column=4, padx=5)

        self.root.bind('<Return>', self.send_message)

        ######### Gelsight =
        self.vs = WebcamVideoStream(src=src).start()
        self.src = src
        self.width = width
        self.height = height
        self.trial = 0
        self.object_class = "0"
        self.frames = 0
        self.recording_active = False
        self.base_image = None
        self.action = "press"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out_record = cv2.VideoWriter("./default.mov", fourcc, CAMERA_CAPTURE_FREQUENCY, (self.height, self.width), isColor=True)
        self.img = None
        self.OF = None
        self.resized_for_OF = False
        self.touch_new = False

        self.image_number = 0
        self.in_demo = True

        # flash out black pixels at the beginning
        self.flash_out_size=20
        self.initialize()

    def initialize(self):
        for i in range(self.flash_out_size):
            self.vs.read()

    def on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def targetCallback(self, msg):
        self.display_message("Not You", msg.data, "left")

        if self.listening_for_objects:
            self.listening_for_objects = False
            items = msg.data.split("\n")
            self.all_items = []
            for item in items:
                self.all_items.append(item.split(", ")[-1][:-1].replace(".", "").strip())
            self.all_items = ",".join(self.all_items)
            print(self.all_items)


    def fabricate_string(self, txt):
        ### this is for direct API calls
        if txt[0] == "$":
            data_split = txt[1:].split("(")
            api_request = data_split[0]
            function = data_split[1][:-1]

            req = {}
            if "describe" in api_request or "rank" in api_request:
                req = {'object_ids': function}
            if api_request == 'reset':
                req = {}
            if api_request == 'ask':
                req = {'query': function}
            if api_request == 'describe_rgb':
                req = {'prompt': function}
            if api_request == "guess_from_objects":
                req = {'object_candidates': function}

            url = 'http://127.0.0.1:8000/' + api_request
            x = requests.post(url, params=req, timeout=30)
            response_dict = ast.literal_eval(x.content.decode("utf-8"))

            print(response_dict)
            response = response_dict  # ['response']
            self.display_message("agent", response)
            return



    def sort_items(self):
        self.fabricate_string("publish sort " + self.rank_criteria)

    def touch_this(self):
        self.display_message("user", "Touch this item.", "right")
        # self.fabricate_string("c")
        self.recording_active = True
        if self.touch_new:
            self.object_class = str(int(self.object_class) + 1)
            if self.object_class == '4':
                print("Be warned, you are adding too many objects!")
                self.object_class = '1'
        self.touch_new = False

        time.sleep(1)
        self.display_message("agent", "Please use the gripper to grab the item.", "left")

        ## TODO integrate your robot gripper control here and make it grasp the object

        file_path_l = '/home/user/Documents/octopi-s/data/demo_videos/demo/{}/item.mov'.format(self.object_class)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out_record = cv2.VideoWriter(file_path_l, fourcc, CAMERA_CAPTURE_FREQUENCY, (self.height, self.width),
                                          isColor=True)

        for i in range(CAMERA_CAPTURE_FREQUENCY * 20):
            img = self.vs.read()
            if self.resized_for_OF:
                self.img = resize_crop_mini(img, self.height, self.width)
            else:
                self.img = cv2.resize(img, (self.height, self.width))

            self.out_record.write(self.img)
            self.frames += 1

            cv2.imshow('Gelsight', self.img)
            if cv2.waitKey(1) == 27:
                break

        self.recording_active = False
        self.frames = 0
        self.out_record = None
        print("Stopping recording for trial " + str(self.trial))

        ## automove to folder 4 too
        shutil.copy('/home/user/Documents/octopi-s/data/demo_videos/demo/{}/item.mov'.format(self.object_class), '/home/user/Documents/octopi-s/data/demo_videos/demo/4/item.mov')
        cv2.destroyAllWindows()

        self.display_message("agent", "Finished collecting data!", "left")

    def touch_new_item(self):
        self.touch_new = True
        self.touch_this()
        # rospy.sleep(20)
        # self.display_message("Them", "Done.", "left")

    def take_pic(self):
        self.display_message("user", "Take a picture of the scene.", "right")

        ### call gopro code to capture image and transfer
        # subprocess.run('bash -c "conda activate goctopi; gopro-photo"', shell=True)
        # downscale_image('photo.jpg', '/home/user/Documents/octopi-s/data/demo_videos/demo/rgb.png')


        ### TODO replace with your own realsense camera image saving and save it to '/home/user/Documents/octopi-s/data/demo_videos/demo/rgb.png'
        
        self.display_message("agent", "Picture taken!", "left")

    def send_message(self):

        message = self.entry_box.get()
        if message.strip() == "":
            return  # Prevent sending empty messages

        if message == "exit":
            self.display_message("Not you", "Goodbye.", "left")
            time.sleep(2)
            self.vs.stop()
            exit()
        # Display user message in a bubble
        self.status_message.set("Octopi is thinking....")
        self.display_message("user", message, "right")

        # Clear the input field
        self.entry_box.delete(0, tk.END)


        try:
            text = message.lower()

            if text[0] == "$":
                self.fabricate_string(text)
                self.status_message.set("")
                return

            if "#" in text:
                # skip to ask command
                url = 'http://127.0.0.1:8000/ask'
                x = requests.post(url, params={'query': text[1:]}, timeout=30)
                response = ast.literal_eval(x.content.decode("utf-8"))
                self.status_message.set("")
                self.display_message("agent", response, "left")
                return

            ###### get_prompt quote here
            req = {'query': text}
            suffix = 'get_response'
            url = 'http://127.0.0.1:8000/' + suffix

            x = requests.post(url, params=req, timeout=30)
            response_dict = ast.literal_eval(x.content.decode("utf-8"))
            print(response_dict['response'])

            if "objects" in response_dict:
                self.all_items = response_dict["objects"]

            self.status_message.set("")
            self.display_message("agent", response_dict["response"], "left")

            return

        except Exception as e:
            self.status_message.set("")
            raise e
            self.display_message("Error", "Error sending message to Octopi: " + str(e), "left")

    def display_message(self, sender, message, side="left"):
        """Create a bubble for a message"""
        # Message bubble container
        message = message.replace("<|eot_id|>", "")
        message = message.replace("<|im_end|>", "")


        bubble_frame = Frame(self.message_frame, bg="#EDEDED", padx=10, pady=5)

        if sender == "user":
            # Align the user message to the right
            bubble_frame.pack(anchor="e", padx=5, pady=5, fill=tk.X, expand=True)
        else:
            # Align the bot response to the left
            bubble_frame.pack(anchor="w", padx=5, pady=5, fill=tk.X, expand=True)

        # Message label inside the bubble
        message_label = tk.Label(bubble_frame, text=message, bg="#DCF8C6" if sender == "user" else "#FFFFFF",
                                 wraplength=500, justify="left", anchor="w", padx=10, pady=5,font=("Helvetica", 20))
        message_label.pack(fill=tk.X, expand=True)
        # if sender != 'user':
        #     self.audio.speak_async(message)

        # Auto-scroll to the bottom after adding a message
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def capture(self):
        img = self.vs.read()
        if self.resized_for_OF:
            self.img = resize_crop_mini(img, self.height, self.width)
        else:
            self.img = cv2.resize(img, (self.width, self.height))

    def capture_display(self):
        self.capture()
        cv2.imshow('Gelsight', self.img)


    def reset_chat(self):
        """Reset the chat area by clearing all messages"""
        for widget in self.message_frame.winfo_children():
            widget.destroy()

        ###### get_prompt quote here
        req = {}
        url = 'http://127.0.0.1:8000/reset/'
        self.object_class = '0'
        x = requests.post(url, params=req, timeout=30)
        # self.fabricate_string("reset")


if __name__ == "__main__":

    # gs_ids = search_for_devices()

    # NUM_SENSORS = len(gs_ids)
    # color = np.random.randint(0, 255, (100, 3))

    # if NUM_SENSORS == 0:
    #     print('No gelsight sensor is found! Exiting ...')
    #     exit()
    # # # run sensors
    # # print(gs_ids)
    # recording_active = False
    
    cv2.destroyAllWindows()
    root = tk.Tk()
    app = Octopi(root, 0, width=WIDTH, height=HEIGHT)
    root.mainloop()
