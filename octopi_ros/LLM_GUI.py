import tkinter as tk
from tkinter import scrolledtext, Canvas, Frame
import requests
import rospy
from std_msgs.msg import String, Float64
import openai
import ast
from gtts import gTTS
import io
import pygame
import threading


### Fill this in with your own chatGPT API key and credentials.
client = openai.OpenAI(
  organization='',
  api_key='',
)


class AudioManager:

    ####
    # Class for playing back the response received.
    ####
    def __init__(self, max_time=10):
        self.max_time = max_time

    def record(self, file_name="my_recording.wav"):
        pass

    def speak(self, text):
        tts = gTTS(text=text, lang='en', tld='us', slow=False)
        # Save the speech to a file-like object in memory (BytesIO)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        self.play(audio_fp)

        # Save the speech to a file
        # audio_fp.seek(0)
        # with open("output.mp3", "wb") as f:
        #     f.write(audio_fp.read())

    def play(self, audio_file):
        # Initialize the mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load(audio_file)

        # Play the audio
        pygame.mixer.music.play()

        # Wait until the music finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(self.max_time)

        pygame.mixer.init()
        sound = pygame.mixer.Sound(audio_file)
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))

    def speak_async(self, text):
        # Run the speak method in a separate thread
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()


class Octopi:
    def __init__(self, root):
        self.root = root
        self.root.title("Octopi")
        self.root.geometry("500x600")

        self.tactile_pub = rospy.Publisher('/gsmini_command', String, queue_size=10)
        rospy.Subscriber('/verbose_output', String, queue_size=1, callback=self.targetCallback)
        self.rank_criteria = ""
        self.listening_for_objects = False
        self.all_items = ""
        self.audio = AudioManager()

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

        # Entry box for typing messages
        self.entry_box = tk.Entry(self.root, width=70)
        self.entry_box.pack(pady=5)

        # Button frame at the bottom
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10, side=tk.BOTTOM)

        # Buttons
        self.touch_items_btn = tk.Button(button_frame, text="Touch Items", width=10, command=self.touch_items)
        self.touch_items_btn.grid(row=0, column=0, padx=5)

        self.touch_items_btn = tk.Button(button_frame, text="Take Picture", width=10, command=self.take_pic)
        self.touch_items_btn.grid(row=0, column=1, padx=5)

        self.touch_this_btn = tk.Button(button_frame, text="Touch This", width=10, command=self.touch_this)
        self.touch_this_btn.grid(row=0, column=2, padx=5)

        self.sort_btn = tk.Button(button_frame, text="Sort", width=10, command=self.sort_items)
        self.sort_btn.grid(row=0, column=3, padx=5)

        self.send_btn = tk.Button(button_frame, text="Send", width=10, command=self.send_message)
        self.send_btn.grid(row=0, column=4, padx=5)

        self.reset_btn = tk.Button(button_frame, text="Reset", width=10, bg='red', fg='white', command=self.reset_chat)
        self.reset_btn.grid(row=0, column=5, padx=5)

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
        msg = String()
        msg.data = txt
        self.tactile_pub.publish(msg)

    def sort_items(self):
        self.fabricate_string("publish sort " + self.rank_criteria)

    def touch_this(self):
        self.display_message("user", "Touch this item.", "right")
        self.fabricate_string("infer")
        # rospy.sleep(7)
        # self.display_message("Them", "Done.", "left")

    def touch_items(self):
        self.display_message("user", "Touch the items on the table.", "right")
        self.fabricate_string("item touch")
        # rospy.sleep(20)
        # self.display_message("Them", "Done.", "left")

    def take_pic(self):
        self.display_message("user", "Take a picture of the scene.", "right")
        self.fabricate_string("take_pic")
        # rospy.sleep(20)
        # self.display_message("Them", "Done.", "left")

    def send_message(self):
        message = self.entry_box.get()
        if message.strip() == "":
            return  # Prevent sending empty messages

        # Display user message in a bubble
        self.display_message("user", message, "right")

        # Clear the input field
        self.entry_box.delete(0, tk.END)

        try:
            text = message.lower()
            #
            prompt = 'You are interfacing with a robot that can sense tactile information and can perform inference on the tactile signals. A user will give a query, and you must identify the most appropriate category that should be sent to the robot. There are 6 categories. Your answer must follow the format: {CATEGORY_NUMBER: "ADDITIONAL DETAILS or NONE if no such details are needed"}\
                        \
                        Category 1: "describe and rank"\
                        Function: Ask the robot to describe the objects in the scene and ranks them by a given criteria, which is either "hardness or roughness". If no such criteria was given, the ADDITIONAL DETAILS will be "uncertain".\
                        Format: {1: "hardness/roughness/uncertain"}\
                        \
                        Category 2: "describe"\
                        Function: Ask the robot to describe the physical properties of one object in the scene. The ADDITIONAL_DETAILS will be 1, 2, or 3 if the user asked for either one of these to be described, otherwise, if it is not certain which object, ADDITIONAL_DETAILS will be 4. If multiple items are to be described, ADDITIONAL_DETAILS should be "1,2,3"\
                        Format: {2: 1/2/3/4}\
                        \
                        Category 3: "rank"\
                        Function: Ask the robot to ranks them by a given criteria, which is either "hardness or roughness". If no such criteria was given, the ADDITIONAL DETAILS will be "uncertain".\
                        Example query: Please rank the items by hardness.\
                        Format: {3: "hardness/roughness/uncertain"}\
                        \
                        Category 4: "guess from objects"\
                        Function: Ask the robot to infers the most likely object given a tactile reading of the object and a list of objects.\
                        Example query: which object is it?\
                        Format: {4: None}\
                        \
                        Category 5: "prompt"\
                        Function: Ask the robot to describe what they see on the table, purely from vision, and does not describe them from tactile feedback.\
                        Format: {5: None}\
                        Category 6: "ask"\
                        Function: This is a catch-all category for queries that do not fulfil any of the above categories. If the query involves some kind of describe, rank, or guessing, it must never be classified under this category.\
                        Example query: The item is not a tennis ball. \
                        Format: {6: None}\
                        \
                        ------------\
                        USER: ' + text + '\
                        \
                        ANSWER:'

            response = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                model="gpt-4o",
            )

            answer = ast.literal_eval(response.choices[0].message.content)
            print(answer)
            answer_type = list(answer.keys())[0]
            context = answer[answer_type]

            ## catch describe and rank
            if answer_type == 1:  # "describe" in text and "rank" in text:
                ### attempt to find it by criteria:
                msg = "describe and rank"
                if context.lower() != "uncertain":
                    self.rank_criteria = context
                # if "hardness" in text:
                #     self.rank_criteria = "hardness"
                # elif "roughness" in text:
                #     self.rank_criteria = "roughness"
                else:
                    self.display_message("Error",
                                         "I could not understand what criteria you want me to sort the items by. Could you try again?",
                                         "left")
                    return
            ## catch describe this item only
            elif answer_type == 2:  # "describe" in text:
                msg = "describe " + str(context)
            ## catch rank only
            elif answer_type == 3:  # "rank" in text:
                msg = "rank"
                if context.lower() != "uncertain":
                    self.rank_criteria = context
                # if "hardness" in text:
                #     self.rank_criteria = "hardness"
                # elif "roughness" in text:
                #     self.rank_criteria = "roughness"
                else:
                    self.display_message("Error",
                                         "I could not understand what criteria you want me to sort the items by. Could you try again?",
                                         "left")
                    return
            ## handling guess from objects (given the list of objects, what is it?)
            elif answer_type == 4:  # "object" in text:
                msg = "guess from objects " + self.all_items

            # elif answer_type == 5:  # "what is this item: " in text:
            #     msg = "guess from touch " + context
            elif answer_type == 5:
                # treat the rest as an ask command.
                msg = "prompt"
                self.listening_for_objects = True
            elif answer_type == 6:
                # treat the rest as an ask command.
                msg = "ask " + text
            else:
                self.display_message("Error",
                                     "Sorry, I could not understand what you are asking for. Could you try again?",
                                     "left")
                return

            self.fabricate_string(msg)
            # self.display_message("Error", msg, "left")
        except Exception as e:
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
                                 wraplength=300, justify="left", anchor="w", padx=10, pady=5)
        message_label.pack(fill=tk.X, expand=True)
        # if sender != 'user':
        #     self.audio.speak_async(message)

        # Auto-scroll to the bottom after adding a message
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)


    def reset_chat(self):
        """Reset the chat area by clearing all messages"""
        for widget in self.message_frame.winfo_children():
            widget.destroy()
        self.fabricate_string("reset")


if __name__ == "__main__":
    root = tk.Tk()
    rospy.init_node("Octopi_tower")
    app = Octopi(root)
    root.mainloop()
