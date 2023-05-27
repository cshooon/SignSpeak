# SignSpeak

## Project Description

"Sign Speak" is an educational software developed using Kivy, a Python library for developing applications. It's a modern and engaging way to learn sign language, offering different interactive modes to enrich the learning experience. The project aims to make sign language more accessible to everyone and promote its learning and usage.

## Features

1. **Word Quiz:** This mode displays a random word to the user in the form of sign language images. The user has to guess the word and type it in. If the guess is correct, it displays a success message; otherwise, it asks the user to try again.

2. **Sign Language:** This mode displays the alphabet and allows the user to click on a letter to view the corresponding sign language image.

3. **ASL Recognition:** This is the advanced mode which uses computer vision and machine learning. It uses a webcam to capture the user's hand movements, predicts the sign language alphabet, and displays it to the user. This mode is useful for practicing sign language and getting instant feedback.

## How to Run and screenshots

To run the software, navigate to the directory where the software is located and run the following command in your terminal:

```python
python main.py
```
1. Word Quiz 
* 3-letter to 5-letter words are expressed in sign language. The order of expressing a letter in sign language is mixed, arrange the word order to guess the word. Click the check button to see if you are right or wrong. If you want to try again, press regenerate button.
<img width="321" alt="worddd" src="https://github.com/cshooon/SignSpeak/assets/113033780/db5c50ba-9bf5-4581-aeeb-9a35f951c2c1">
<img width="321" alt="AAA" src="https://github.com/cshooon/SignSpeak/assets/113033780/7bed566d-259f-45d0-8d33-90cb3b98296d">

2. Sign Language
* sign language of each alphabet(A to Z)
<img width="321" alt="HH" src="https://github.com/cshooon/SignSpeak/assets/113033780/588bdf2f-e614-4430-bad3-c014897f2f4a">
<img width="321" alt="tt" src="https://github.com/cshooon/SignSpeak/assets/113033780/95389ee4-98af-4093-826d-83080c1233fc">


3. ASL Recognition
* First, press Start Camera button. Then show your hand!! I don't know the exact reason, but it's not accurate. I assume it's because the dataset is small.
<img width="321" alt="C" src="https://github.com/cshooon/SignSpeak/assets/113033780/c15f3135-7c75-45f3-964a-202bb2e1cb1d">
<img width="321" alt="P" src="https://github.com/cshooon/SignSpeak/assets/113033780/a00dde1a-74c8-4a74-aab5-e482b7f4000d">


## Requirements

This project uses the following libraries and modules:

- Kivy
- OpenCV
- NLTK
- MediaPipe
- Tensorflow
- Numpy

Also, download the required NLTK data:
```python
import nltk
nltk.download('brown')
```

## Troubleshooting
1. hand tracking(find hand)
This class, `handTracker`, is responsible for detecting and tracking hands in an image using the MediaPipe library. There are its two methods.

    1. **`handsFinder`**: This method detects the hand landmarks in the given image. 

    2. **`positionFinder`**: This method returns a list of all hand landmarks, with each landmark being a list of its ID and its x and y coordinates.
    
```python
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
```
2. convert sign language to Alphabet

     1. **`Hand Region Isolation`**: The wrist and middle finger's coordinates are used as an approximate bounding box for the hand.

    2. **`Hand Region Preprocessing`**: The isolated hand region is then preprocessed to match the requirements of  model. The image is resized to a fixed size (64x64 pixels) which is the input size expected by the model. It's then normalized (all pixel values divided by 255) to bring all pixel intensities within the range [0, 1]. The normalized image is then expanded along a new axis to match the model's expected input shape.

    3. **`Alphabet Prediction`**: Lastly put the preprocessed hand image into the trained model, which then predicts the ASL (American Sign Language) alphabet corresponding to the hand posture. The model's output is a vector of probabilities corresponding to each potential alphabet. The alphabet with the highest probability is selected as the predicted output.

```python
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
```
## Future Enhancements
I couldn't deploy because depolyment to ios requires macos. (I'm using window 😞😞) During summer vacation I'll try to deploy with android. I've tried to improve trained model to be well fitted using pretained model in tensorflow. Despite my effort, my computer RAM can't afford it. :cry::cry::upside_down_face: 

During summer vacation I'll try with my future new computer. 🙂:smile:

## References
* [ASL training model](https://www.kaggle.com/code/namanmanchanda/asl-detection-99-accuracy)

