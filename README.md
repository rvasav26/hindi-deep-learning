<img width="800" alt="Screenshot 2024-09-27 at 9 48 59 PM" src="https://github.com/user-attachments/assets/58fb234b-af1f-4a04-8051-ea5197e475bc">

# Hindi Deep Learning Scratchpad and Airpad
Used dataset of 9,000+ handwritten Devanagari characters to develop and train a TensorFlow convolutional neural network (CNN). 

## Part I: Scratchpad
Using OpenCV and NumPy, I developed a program to enable users to write characters on an online scratchpad. The CNN recieves the image of the handwritten text in the form of a NumPy array and, after
performing matrix transformations, makes a prediction of the character. This process is entirely in real-time, and the user receives feedback from the model with a latency under 100 milliseconds.
Maximum accuracy achieved was 93% (correct predictions of characters/total predictions)

## Part II: Airpad
This is an extension to Part I, where, instead of drawing with a mouse, users can draw in mid-air with their pointer finger. This is done by using the additional MediaPipe library. First, 
<img width="670" alt="Screenshot 2025-04-05 at 1 56 32 PM" src="https://github.com/user-attachments/assets/4c3e7bfb-37f9-45e0-a8f5-3487e5e7b24f" />

## Links
Video Demo: https://www.youtube.com/watch?v=B65aY0wFP3U
