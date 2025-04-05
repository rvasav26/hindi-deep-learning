<img width="650" alt="img1" src="https://github.com/user-attachments/assets/58fb234b-af1f-4a04-8051-ea5197e475bc">

# Hindi Deep Learning Scratchpad and Airpad
Used dataset of 9,000+ handwritten Devanagari characters to develop and train a TensorFlow convolutional neural network (CNN). Applied CNN to two <a href="https://www.youtube.com/watch?v=K-BgNTboKrQ">applications</a> below:

## Part I: Scratchpad
Using OpenCV and NumPy, I developed a program to enable users to write characters on an online scratchpad. The CNN recieves the image of the handwritten text in the form of a NumPy array and, after
performing matrix transformations, makes a prediction of the character. This process is entirely in real-time, and the user receives feedback from the model with a latency under 100 milliseconds.
Maximum accuracy achieved was 93% (correct predictions of characters/total predictions)

## Part II: Airpad

<img width="500" alt="img2" src="https://github.com/user-attachments/assets/4c3e7bfb-37f9-45e0-a8f5-3487e5e7b24f">

This is an extension to Part I, where, instead of drawing with a mouse, users can draw in mid-air with their pointer finger. This is done by using the additional MediaPipe library. First, the user's hand and pointer finger are detected and their locations saved. Second, a face mesh is applied to the user's face and two locations are noted: namely, the upper and lower inner lip. This second step is to allow the user to "put down" their finger pen (if their mouth is closed, they are drawing, and if their mouth is open, they are not drawing), to prevent them from drawing when they don't want to. Then, the user's drawing is transformed into an array that the CNN can interpet. Finally, the CNN outputs its prediction of the Hindi letter the user has drawn. This is all done in real-time.

## Links
<a href="https://www.youtube.com/watch?v=K-BgNTboKrQ">Video Demo for Scratchpad</a>

<a href="https://www.youtube.com/watch?v=B65aY0wFP3U">Video Demo for Scratchpad and Airpad (Enhanced Model)</a>
