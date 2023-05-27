from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.camera import Camera
import nltk
from nltk.corpus import brown
import random
import time
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp
from tensorflow import keras
import numpy as np


class QuizScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layout = BoxLayout(orientation='vertical')  

        # Add a "Word Quiz" label
        self.app_name_label = Label(text='Word Quiz', size_hint=(1, 0.2), font_size='20sp')
        self.layout.add_widget(self.app_name_label)

        nltk.download('brown')
        self.common_words = [word for word in brown.words() if 3 <= len(word) <= 5]

        self.current_word = None
        self.images = None

        self.box = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.layout.add_widget(self.box)

        self.reset_word()

        # Add an input field for guessing
        self.button_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
        self.user_input = TextInput(multiline=False)
        self.button_box.add_widget(self.user_input)
        self.layout.add_widget(self.button_box)

        # Add a "Check" button
        self.below_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
        self.check_button = Button(text='Check')
        self.check_button.bind(on_press=self.check_word)
        self.below_box.add_widget(self.check_button)

        # Add a "Help" button
        self.help_button = Button(text='Help')
        self.help_button.bind(on_press=self.display_help)
        self.below_box.add_widget(self.help_button)
        self.layout.add_widget(self.below_box)

        self.regenerate_button = Button(text='Regenerate',
                                        size_hint=(1, 1),
                                        background_color=(0.9, 0.9, 0.9, 1),  # Light grey color
                                        color=(0, 0, 0, 1))  # Black text
        self.regenerate_button.bind(on_press=self.reset_word)

        self.back_button = Button(text='Back',
                                  size_hint=(1, 1),
                                  background_color=(0.9, 0.9, 0.9, 1),  # Light grey color
                                  color=(0, 0, 0, 1))  # Black text
        self.back_button.bind(on_press=self.go_back)

        self.bottom_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.bottom_box.add_widget(self.regenerate_button)
        self.bottom_box.add_widget(self.back_button)
        self.layout.add_widget(self.bottom_box)

        self.add_widget(self.layout)

    def reset_word(self, *args):
        self.box.clear_widgets()
        self.current_word = random.choice(self.common_words)
        self.images = self.create_images()
        random.shuffle(self.images)

        for i in range(5):
            if i < len(self.images):
                self.box.add_widget(self.images[i])
            else:
                self.box.add_widget(Button(background_color=(0, 0, 0, 0), disabled=True))

    def create_images(self):
        image_buttons = []
        for letter in self.current_word:
            btn = Button(
                background_normal=f'alphabet/{letter.lower()}_test.jpg')
            image_buttons.append(btn)
        return image_buttons

    def check_word(self, instance):
        if self.user_input.text.lower() == self.current_word:
            print("Correct!")
            self.check_button.text = 'O'
        else:
            print("Incorrect. Try again.")
            self.check_button.text = 'X'

    def display_help(self, instance):
        print("Help information: ...")

    def go_back(self, instance):
        self.manager.current = 'main'


class SignLanguageScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.button_states = {}
        self.button_originals = {}
        layout = GridLayout(cols=3)
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            btn = Button(text=letter)
            btn.bind(on_press=self.change_image)
            layout.add_widget(btn)
            self.button_states[btn] = "text"
            self.button_originals[btn] = (btn.background_normal, btn.text)
        back_button = Button(text='Back',
                             size_hint=(1, 0.1),
                             pos_hint={'x': 0, 'y': 0},
                             background_color=(0.9, 0.9, 0.9, 1),
                             color=(0, 0, 0, 1))
        back_button.bind(on_press=self.go_back)
        layout.add_widget(back_button)
        self.add_widget(layout)

    def change_image(self, instance):
        if self.button_states[instance] == "text":
            instance.background_normal = f'alphabet/{instance.text}_test.jpg'
            instance.text = ''  # Remove the text
            self.button_states[instance] = "image"  # change state to image
        else:
            instance.background_normal, instance.text = self.button_originals[instance]
            self.button_states[instance] = "text"  # change state back to text

    def go_back(self, instance):
        self.manager.current = 'main'



class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist

class KivyCamera(Image):
    def __init__(self, asl_screen, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.asl_screen = asl_screen
        self.model = keras.models.load_model('model/ASL.h5')
        self.capture = capture
        self.tracker = handTracker()
        self.fps = fps
        self.camera_on = False

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = self.tracker.handsFinder(frame)
            lmList = self.tracker.positionFinder(frame)
            if len(lmList) != 0:
                # Get the wrist and middle fingertip landmarks
                wrist_x, wrist_y = lmList[0][1], lmList[0][2]  # Wrist is landmark 0
                mid_tip_x, mid_tip_y = lmList[12][1], lmList[12][2]  # Middle finger tip is landmark 12

                # Define the bounding box with padding
                p = 20
                x_min, y_min = max(0, min(wrist_x, mid_tip_x) - p), max(0, min(wrist_y, mid_tip_y) - p)
                x_max, y_max = min(frame.shape[1], max(wrist_x, mid_tip_x) + p), min(frame.shape[0],
                                                                                     max(wrist_y, mid_tip_y) + p)

                # Crop the hand region from the frame
                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    return

                # Resize the hand image to (64, 64)
                hand_img = cv2.resize(hand_img, (64, 64))

                # Preprocess the hand image
                hand_img = hand_img.astype('float32') / 255.0

                # Add an extra dimension
                hand_img = np.expand_dims(hand_img, axis=0)

                # Predict the ASL alphabet
                pred = self.model.predict(hand_img)
                pred_class = np.argmax(pred, axis=1)

                asl_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                                'del', 'nothing', 'space']
                predicted_char = asl_alphabet[pred_class[0]]

                ASLRecognitionScreen.update_prediction_label(self.asl_screen, predicted_char)
                print(f'Predicted ASL Alphabet: {predicted_char}')

            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture

    def start_camera(self):
        if not self.camera_on:
            Clock.schedule_interval(self.update, 1.0 / self.fps)
            self.camera_on = True

    def stop_camera(self):
        if self.camera_on:
            Clock.unschedule(self.update)
            self.camera_on = False


class ASLRecognitionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = 0  # Camera index. Change this to switch between cameras

        layout = FloatLayout()

        # Create a KivyCamera object
        self.cam = KivyCamera(self, capture=cv2.VideoCapture(0), fps=30)
        self.cam.size_hint = (1, 0.8)
        self.cam.pos_hint = {'top': 1}
        layout.add_widget(self.cam)

        # Add a label to display the predicted character
        self.prediction_label = Label(size_hint=(1, 0.05), pos_hint={'x': 0, 'y': 0.15})
        layout.add_widget(self.prediction_label)

        # Add a button to start/stop the camera
        cam_button = Button(text='Start Camera', size_hint=(0.5, 0.05), pos_hint={'x': 0, 'y': 0.05})
        cam_button.bind(on_press=self.switch_cam)
        layout.add_widget(cam_button)

        # Add a button to switch the camera
        switch_button = Button(text='Switch Camera', size_hint=(0.5, 0.05), pos_hint={'x': 0.5, 'y': 0.05})
        switch_button.bind(on_press=self.switch_cam_index)
        layout.add_widget(switch_button)

        # Add a button to capture the photo
        capture_button = Button(text='Capture', size_hint=(1, 0.05), pos_hint={'x': 0, 'y': 0.1})
        capture_button.bind(on_press=self.capture)
        layout.add_widget(capture_button)

        # Add a button to go back
        back_button = Button(text='Back', size_hint=(1, 0.05), pos_hint={'x': 0, 'y': 0})
        back_button.bind(on_press=self.go_back)
        layout.add_widget(back_button)

        self.add_widget(layout)

    def switch_cam(self, instance):
        if self.cam.camera_on:
            self.cam.stop_camera()
            instance.text = 'Start Camera'
        else:
            self.cam.start_camera()
            instance.text = 'Stop Camera'

    def switch_cam_index(self, instance):
        try:
            new_index = 0 if self.cam.index else 1
            new_cam = Camera(index=new_index)
            self.cam.index = new_index
        except Exception as e:
            print(f"Camera with index {new_index} is not available: {e}")

    def capture(self, instance):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.cam.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")

    def update_prediction_label(self, prediction):
        self.prediction_label.text = f"Predicted ASL Alphabet: {prediction}"

    def go_back(self, instance):
        self.manager.current = 'main'




class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        app_name_label = Label(text='Sign Speak', size_hint=(1, 0.3),
                               pos_hint={'x': 0, 'y': 0.7}, font_size="30sp")
        layout.add_widget(app_name_label)
        quiz_button = Button(text='Start Quiz',
                             size_hint=(0.6, 0.1),
                             pos_hint={'center_x': 0.5, 'center_y': 0.6},
                             background_color=(0.9, 0.9, 0.9, 1),
                             color=(0, 0, 0, 1))
        quiz_button.bind(on_press=self.start_quiz)

        sign_button = Button(text='Sign Language',
                             size_hint=(0.6, 0.1),
                             pos_hint={'center_x': 0.5, 'center_y': 0.4},
                             background_color=(0.9, 0.9, 0.9, 1),
                             color=(0, 0, 0, 1))
        sign_button.bind(on_press=self.start_sign_language)

        recognition_button = Button(text='ASL recognition',
                                    size_hint=(0.6, 0.1),
                                    pos_hint={'center_x': 0.5, 'center_y': 0.2},
                                    background_color=(0.9, 0.9, 0.9, 1),
                                    color=(0, 0, 0, 1))
        recognition_button.bind(on_press=self.recognition)

        layout.add_widget(quiz_button)
        layout.add_widget(sign_button)
        layout.add_widget(recognition_button)
        self.add_widget(layout)

    def start_quiz(self, instance):
        self.manager.current = 'start_quiz'

    def start_sign_language(self, instance):
        self.manager.current = 'sign_language'

    def recognition(self, instance):
        self.manager.current = 'asl_recognition'

class MyApp(App):
    def build(self):
        # Set the window size to iPhone 13
        Window.size = (390, 844)
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(QuizScreen(name='start_quiz'))
        sm.add_widget(SignLanguageScreen(name='sign_language'))
        sm.add_widget(ASLRecognitionScreen(name='asl_recognition'))

        return sm

if __name__ == '__main__':
    MyApp().run()
