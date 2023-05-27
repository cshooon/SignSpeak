# SignSpeak

## Project Description

"Sign Speak" is an educational software developed using Kivy, a Python library for developing multitouch applications. It's a modern and engaging way to learn sign language, offering different interactive modes to enrich the learning experience. The project aims to make sign language more accessible to everyone and promote its learning and usage.

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
