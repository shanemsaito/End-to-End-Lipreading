# End-to-End-Lipreading

This project is based on the 2016 paper [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599).

## Model Summary

This model uses a subset of the GRID corpus dataset, consisting of 3-second videos @ 25 fps. Using a combination of Spatial-temporal Convolutional Neural Networks and bidirectional LSTMs, the model can make sentence-level predictions of the testing data set at 95.2% accuracy.

## Dataset

The GRID corpus dataset consists of 34000 total videos, each with a length of 3 seconds @ 25 fps. There are 34 speakers (18 male, 16 female), each with 1000 sentences spoken. Sentences are of the form "put red at G9 now". This model uses a subset of 28775 videos for training and 3971 videos for testing. Each video has a respective alignment which stores each word in the sentence with its respective time stamp. 

Example Video:


Example Alignment

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

Afterwards, 

## Model


## Results
