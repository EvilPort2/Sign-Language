# Sign-Language
A very simple CNN project.

## What I did here
1. The first thing I did was, I created 10 gesture samples using OpenCV. For each gesture I captured 1200 images which were 30x30 pixels. All theses images were in grayscale which is stored in the gestures/ folder. The gestures/0/ folder contains 1200 blank images which signify "none" gesture. Also I realised that keeping this category increased my model's accuracy to 99% from a laughable 82%.
2. Learned what a CNN is and how it works. Best resources were <a href="https://www.tensorflow.org/get_started/">Tensorflow's official website</a> and <a href="https://machinelearningmastery.net">machinelearningmastery.net</a>.
3. Created a CNN which look a lot similar to <a href="https://www.tensorflow.org/tutorials/layers">this MNIST classifying model</a> using both Tensorflow and Keras. If you want to add more gestures you might need to add your own layers and also tweak some parameters, that you have to do on your own.
4. Then used the model which was trained using Keras on a video stream.

There are a lot of details that I left. But these are the basic and main steps.

## Requirements
0. Python 3.x
1. <a href="https://tensorflow.org">Tensorflow 1.5</a>
2. <a href="https://keras.io">Keras</a>
3. OpenCV 3.4
4. h5py
5. A good grasp over the above 5 topics along with neural networks. Refer to the internet if you have problems with those. I myself am just a begineer in those.
6. A good CPU (preferably with a GPU).
7. Patience.... A lot of it.

## How to use this repo
Before using this repo, let me warn about something. You will have no interactive interface that will tell you what to do. So you will have to figure out most of the stuff by yourself and also make some changes to the scripts if the needs arise. But here is a basic gist.

### Creating a gesture
  1. First set your hand histogram. You do not need to do it again if you have already done it. But you do need to do it if the lighting conditions change. To do so type the command given below and follow the instructions 2-9 <a href="https://github.com/EvilPort2/Simple-OpenCV-Calculator/blob/master/README.md">here</a>.
    
    python set_hand_hist.py
  2. The next thing you need to do is create your gestures. That is done by the command given below. On starting executing this program, you will have to enter the gesture number and gesture name/text. Since no checks are implemented here I suggest you do this carefully. Then an OpenCV window called "Capturing gestures" which will appear. In the webcam feed you will see a green window (inside which you will have to do your gesture) and a counter that counts the number of pictures stored.

    python create_gestures.py    
3. Press 'c' when you are ready with your gesture. Capturing gesture will begin after a few seconds. Move your hand a little bit here and there. After the counter reaches 1200 the window will close automatically.
  4. When you are done adding new gestures run the load_images.py file once. You do not need to run this file again until and unless you add a new gesture.
    
    python load_images.py
5. Do not forget to update the num_of_classes variable in cnn_tf.py and cnn_keras.py file if you add any new gestures.

### Training a model
  1. So training can be done with either Tensorflow or Keras. If you want to train using Tensorflow then run the cnn_tf.py file. If you want to train using Keras then use the cnn_keras.py file.
  
    python cnn_tf.py
    python cnn_keras.py
2. If you use Tensorflow you will have the checkpoints and the metagraph file in the tmp/cnn_model3 folder.
3. If you use Keras you will have the model in the root directory by the name cnn_keras2.h5.

### Recognizing gestures
Before going into much details I would like to tell that I was not able to use the model trained using tensorflow. That is because I do not know how to use it. I tried using the predict() function of the Estimator API but that loads the parameters into memory every time it is called which is a huge overhead. Please help me if you can with this. The functions for prediction using tf is tf_predict() which you will find in the recognize_gesture.py file but it is never used.
This is why I ended up using Keras' model, as the loading the model into memory and using it for prediction is super easy.
  1. For recognition start the recognize_gesture.py file.

    python recognize_gesture.py
2. You will have a small green box inside which you need to do your gestures.

# Got a question?
If you have any questions that are bothering you please contact me on my <a href = "http://www.facebook.com/dibakar.saha.750">facebook profile</a>. Just do not ask me questions like where do I live, who do I work for etc. Also no questions like what does this line do. If you think a line is redundant or can be removed to make the program better then you can obviously ask me or make a pull request.
