# End-to-End-Lipreading

This project is based on the 2016 paper [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599).

## Model Summary

This model uses a subset of the GRID corpus dataset, consisting of 3-second videos @ 25 fps. Using a combination of Spatial-temporal Convolutional Neural Networks and bidirectional GRUs, the model can make sentence-level predictions of the testing data set at 95.2% accuracy.

## Dataset

The GRID corpus dataset consists of 34000 total videos, each with a length of 3 seconds @ 25 fps. There are 34 speakers (18 male, 16 female), each with 1000 sentences spoken. Sentences are of the form "put red at G9 now". This model uses a subset of 28775 videos for training and 3971 videos for testing. Each video has a respective alignment which stores each word in the sentence with its respective time stamp. 

Example Video:

![](https://github.com/shanemsaito/end-to-end-lipreading/blob/main/bbaf2n-ezgif.com-video-to-gif-converter.gif)

Example Alignment:

```
0 23750 sil
23750 29500 bin
29500 34000 blue
34000 35500 at
35500 41000 f
41000 47250 two
47250 53000 now
53000 74500 sil
```

To prepare the data, we convert each video to greyscale and normalize the frames to improve data consistency and model performance. 

```python3
    def load_video(path:str) -> List[float]:

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        #Cutting out the mouth portion of the video
        frames.append(frame[190:236,80:220,:])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
```

Afterwards, we convert the character alignments to numbers and pad them with 0's if they are less than 40 tokens to make them all the same length. 

## Model

The model is represented by the diagram below: 

![](https://github.com/shanemsaito/end-to-end-lipreading/blob/main/cd8d0bfdef7ca5065ad735fb0afc8a49.png)

The frames feed through three spatial-temporal convolutions (activated by a ReLU) each followed by a spatial max pooling layer. Then, the extracted features are processed by 2 Bidirectional GRUs, in which each step is processed by a linear layer and a softmax. This model uses Connectionist Temporal Classification (CTC) for loss so it can apply to datasets without an alignment transcription. 

### Implemented Model

```python
model = Sequential()

#Spatial-temporal Convolutional Neural Networks
model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

#Flattening tensors before the Bidirectional GRUs
model.add(TimeDistributed(Flatten()))

#Bidrectional GRUs
model.add(Bidirectional(GRU(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(GRU(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

#Dense Layer
model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))
```

### CTC Loss Function

```python
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss
```
