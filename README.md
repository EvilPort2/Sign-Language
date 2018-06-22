[![DOI](https://zenodo.org/badge/122105084.svg)](https://zenodo.org/badge/latestdoi/122105084)

# Sign-Language
A very simple CNN project.

## Note
Simple-OpenCV-Calculator and this project are merged to one. Simple-OpenCV-Calculator will no longer be maintained.

## What I did here
1. The first thing I did was, I created 44 gesture samples using OpenCV. For each gesture I captured 1200 images which were 50x50 pixels. All theses images were in grayscale which is stored in the gestures/ folder. The pictures were flipped using flip_images.py. This script flips every image along the vertical axis. Hence each gesture has 2400 images.
2. Learned what a CNN is and how it works. Best resources were <a href="https://www.tensorflow.org/get_started/">Tensorflow's official website</a> and <a href="https://machinelearningmastery.com">machinelearningmastery.com</a>.
3. Created a CNN which look a lot similar to <a href="https://www.tensorflow.org/tutorials/layers">this MNIST classifying model</a> using both Tensorflow and Keras. If you want to add more gestures you might need to add your own layers and also tweak some parameters, that you have to do on your own.
4. Then used the model which was trained using Keras on a video stream.
5. As of today, I have stored the 44 gestures for which are 26 alphabets and 10 numbers of American Sign language and some other gestures. And trained the model on these images.

There are a lot of details that I left. But these are the basic and main steps.

## Outcome
Watch it <a href="https://youtu.be/JNZ7oFaH1fg">here</a>.

## Requirements
0. Python 3.x
1. <a href="https://tensorflow.org">Tensorflow 1.5</a>
2. <a href="https://keras.io">Keras</a>
3. OpenCV 3.4
4. h5py
5. pyttsx3
6. A good grasp over the above 5 topics along with neural networks. Refer to the internet if you have problems with those. I myself am just a begineer in those.
7. A good CPU (preferably with a GPU).
8. Patience.... A lot of it.

## Installing the requirements
1. Start your terminal of cmd depending on your os.
  2. If you have a NVidia GPU then make sure you have the prerequisites for Tensorflow GPU installation (Refer to official site). Then use this commmand

    pip install -r requirements_gpu.txt

  3. In case you do not have a GPU then use this command

    pip install -r requirements_cpu.txt

## How to use this repo
Before using this repo, let me warn about something. You will have no interactive interface that will tell you what to do. So you will have to figure out most of the stuff by yourself and also make some changes to the scripts if the needs arise. But here is a basic gist.

### Creating a gesture 
  0. Watch the video guide for setting the hand histogram <a href='https://youtu.be/KYfBLeYDMW4'>here</a>.
  1. First set your hand histogram. You do not need to do it again if you have already done it. But you do need to do it if the lighting conditions change. To do so type the command given below and follow the instructions below.
    
    python set_hand_hist.py

  * A windows "Set hand histogram" will appear.
  * "Set hand histogram" will have 50 squares (5x10).
  * Put your hand in those squares. Make sure your hand covers all the squares.
  * Press 'c'. 1 other window will appear "Thresh".
  * On pressing 'c' only white patches corresponding to the parts of the image which has your skin color should appear on the "Thresh" window. 
  * Make sure all the squares are covered by your hand.
  * In case you are not successful then move your hand a little bit and press 'c' again. Repeat this until you get a good histogram.
  * After you get a good histogram press 's' to save the histogram. All the windows close.
  
  2. I already have added 44 (0-43) gestures. It is on you if you want to add even more gestures or replace my gestures. Hence this step is <b>OPTIONAL</b>. To create your own gestures or replace my gestures do the following. It is done by the command given below. On starting executing this program, you will have to enter the gesture number and gesture name/text. Then an OpenCV window called "Capturing gestures" which will appear. In the webcam feed you will see a green window (inside which you will have to do your gesture) and a counter that counts the number of pictures stored.

    python create_gestures.py   

  3. Press 'c' when you are ready with your gesture. Capturing gesture will begin after a few seconds. Move your hand a little bit here and there. You can pause capturing by pressing 'c' and resume it by pressing 'c'. Capturing resumes after a few secondAfter the counter reaches 1200 the window will close automatically.

  After capturing all the gestures you can flip the images using

    python flip_images.py

  4. When you are done adding new gestures run the load_images.py file once. You do not need to run this file again until and unless you add a new gesture.
    
    python load_images.py

### Displaying all gestures
  1. To see all the gestures that are stored in 'gestures/' folder run this command
    
    python display_all_gestures.py

### Training a model
  1. So training can be done with either Tensorflow or Keras. If you want to train using Tensorflow then run the cnn_tf.py file. If you want to train using Keras then use the cnn_keras.py file.
  
    python cnn_tf.py
    python cnn_keras.py
2. If you use Tensorflow you will have the checkpoints and the metagraph file in the tmp/cnn_model3 folder.
3. If you use Keras you will have the model in the root directory by the name cnn_model_keras2.h5.

You do not need to retrain your model every time. In case you added or removed a gesture then you need to retrain it.

### Get model reports
  1. To get the classification reports about the model make sure you have test_images and test_labels file which are generated by load_images.py. In case you do not have them run load_images.py file again. Then run this file

    python get_model_reports.py
  2. You will get the confusion matrix, f scores, precision and recall for the predictions by the model.

### Testing gestures
Before going into much details I would like to tell that I was not able to use the model trained using tensorflow. That is because I do not know how to use it. I tried using the predict() function of the Estimator API but that loads the parameters into memory every time it is called which is a huge overhead. Please help me if you can with this. The functions for prediction using tf is tf_predict() which you will find in the recognize_gesture.py file but it is never used.
This is why I ended up using Keras' model, as the loading the model into memory and using it for prediction is super easy.
   1. First set your hand histogram. 0. Watch the video guide for setting the hand histogram <a href='https://youtu.be/KYfBLeYDMW4'>here</a>. You do not need to do it again if you have already done it. But you do need to do it if the lighting conditions change. To do so type the command given below and follow the instructions below.
    
    python set_hand_hist.py

  * A windows "Set hand histogram" will appear.
  * "Set hand histogram" will have 50 squares (5x10).
  * Put your hand in those squares. Make sure your hand covers all the squares.
  * Press 'c'. 1 other window will appear "Thresh".
  * On pressing 'c' only white patches corresponding to the parts of the image which has your skin color should appear on the "Thresh" window. 
  * Make sure all the squares are covered by your hand.
  * In case you are not successful then move your hand a little bit and press 'c' again. Repeat this until you get a good histogram.
  * After you get a good histogram press 's' to save the histogram. All the windows close.
  2. For recognition start the recognize_gesture.py file.

    python recognize_gesture.py
3. You will have a small green box inside which you need to do your gestures.

### Using fun_util.py
Here is where you will have all the fun. 
  1. First set your hand histogram. You do not need to do it again if you have already done it. But you do need to do it if the lighting conditions change. To do so type the command given below and follow the instructions below.
    
    python set_hand_hist.py

  * A windows "Set hand histogram" will appear.
  * "Set hand histogram" will have 50 squares (5x10).
  * Put your hand in those squares.
  * Press 'c'. 2 other windows will appear. "res" and "Thresh".
  * On pressing 'c' only the parts of the image which has your skin color should appear on the "res" window. White patches corresponding to this should appear on the "Thresh" window. 
  * In case you are not successful then move your hand a little bit and press 'c' again. Repeat this until you get a good histogram.
  * After you get a good histogram press 's' to save the histogram. All the windows close.
  
  2. Start the file.
  
    python fun_util.py

#### Text Mode (Press 't' to go to text mode)
1. In text mode you can create your own words using fingerspellings or use the predefined gestures.
2. The text on screen will be converted to speech on removing your hand from the green box
3. Make sure you keep the same gesture on the green box for 15 frames or else the gesture will not be converted to text.

#### Calculator Mode (Press 'c' to go to calculator mode)
1. To confirm a digit make sure you keep the same gesture for 20 frames. On successful confirmation, the number will appear in the vertical center of the black part of the window.
2. To confirm a number make the "best of luck" gesture and keep in the green box for 25 frames. You will get used to the timing :P.
3. You can have any number of digits for both first number and second number.
4. Currently there are 10 operators.
5. During operator selection, 1 means '+', 2 means '-', 3 means '\*', 4 means '/', 5 means '%', 6 means '\*\*', 7 means '>>' or right shift operator, 8 means '<<' or left shift operator, 9 means '&' or bitwise AND and 0 means '|' or bitwise OR.


# Got a question?
If you have any questions that are bothering you please contact me on my <a href = "http://www.facebook.com/dibakar.saha.750">facebook profile</a>. Just do not ask me questions like where do I live, who do I work for etc. Also no questions like what does this line do. If you think a line is redundant or can be removed to make the program better then you can obviously ask me or make a pull request.

# How to cite 
Saha, D.. (2018, May 9). Sign-Language (Version 1). figshare. https://doi.org/10.6084/m9.figshare.6241901.v1A very simple CNN project.
