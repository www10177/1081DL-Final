audio classification
use ESC-50 dataset (https://github.com/karolpiczak/ESC-50) 
    a labeled collection of 2000 environmental audio recordings .The dataset consists of 5-second-long recordings organized into 50 semantical classes.

1. Augment audio data
    we use the following four methods to get more training data 
    a. add white noise 
    b. shift sound in timeframe 
    c. stretch sound 
    d. combine the above method
    and calculate these audio recordings by Inverse short-time Fourier transform. Then calculate Mel-scaled spectrogram through the above results


2. Define Keras CNN
    Activation function : relu , softmax
    Optimization and callbacks
        optimatizer : Adam

3. prediction and evaluation
