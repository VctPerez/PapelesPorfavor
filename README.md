# PapelesPorfavor
PapelesPorfavor is a Computer Vision Project made by Victor Perez Armenta

This project is based on the famous game called [Papers please](https://papersplea.se/).
The main tools which are going to be used are: [OCR pytesseract](https://pypi.org/project/pytesseract), 
[Face-Recognition](https://pypi.org/project/face-recognition), [OpenCV](https://opencv.org), and
[PyQt5](https://pypi.org/project/PyQt5)
![alt](https://media.tenor.com/uERz3aBsbcAAAAAC/jacksepticeye-papers-please.gif)


## Purpose
Firstly, its main purpose is going to be to recognise failures in passports, like name, country, etc.
I would like to implement something about facial recognition too, but I will go for it just if I get the time (I could).

## Language 
Python will be the main programming language in this project, in order to get experience coding with it.

## How it works?
The app gives you two options: create a passport or submit it.
* Create a Passport: You have to introduce all the data the app asks you to. Then, you can save the passport, 
or reset it.
  * Name: Doesn't need to be on a specified way.
  * Birthdate: It must be **dd/mm/yyyy**.
  * Gender: Your choice :) .
  * Issuer: It has to be on the issuer's list.
  * Expiration date: It must be **dd/mm/yyyy**.
  * Identifier: It must have an "-" splitting it in two words.
  * Picture: Obviously, it must be yours.
  

* Submit a Passport: You will have to choose the passport you want to use, and the app will check if it's okay:
  * Name: It mustn't be in the criminal list.
  * Birthdate: You must be, at least, 18 years old.
  * Gender: It's not checked.
  * Issuer: It must be in the issuers list.
  * Expiration date: It mustn't be expired.
  * Identifier: It checks the previous condition.
  * Picture: You must be the person who did the passport

## Installation
1. Choose your favourite python package manager.
2. Install Python, opencv, PyQt5, Face-Recognition, PyTesseract, matplotlib (in case you want to compile *checkers.py*), skimage and scipy.
3. Download the [Tesseract Google OCR Engine](https://github.com/UB-Mannheim/tesseract/wiki) compatible version with your OS.
4. In **functions.py** replace the *tesseract_cmd* path with your *tesseracct.exe* path
5. Run main.py in cmd, or any development environment.
6. Follow the previous instructions and... **ENJOY!** 