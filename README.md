# SignSpeak

## Project Description

"Sign Speak" is an educational software developed using Kivy, a Python library for developing multitouch applications. It's a modern and engaging way to learn sign language, offering different interactive modes to enrich the learning experience. The project aims to make sign language more accessible to everyone and promote its learning and usage.

## Features

1. **Word Quiz:** This mode displays a random word to the user in the form of sign language images. The user has to guess the word and type it in. If the guess is correct, it displays a success message; otherwise, it asks the user to try again.

2. **Sign Language:** This mode displays the alphabet and allows the user to click on a letter to view the corresponding sign language image.

3. **ASL Recognition:** This is the advanced mode which uses computer vision and machine learning. It uses a webcam to capture the user's hand movements, predicts the sign language alphabet, and displays it to the user. This mode is useful for practicing sign language and getting instant feedback.

## How to Run

To run the software, navigate to the directory where the software is located and run the following command in your terminal:

```python
python main.py
```

## Requirements

This project uses the following libraries and modules:

- Kivy
- OpenCV
- NLTK
- MediaPipe
- Tensorflow
- Numpy

Make sure to install them using pip:

```python
pip install -r requirements.txt
```

Also, download the required NLTK data:
```python
import nltk
nltk.download('brown')
```
