# End-to-end Dog Breed Classification
Using Transfer Learning and TensorFlow 2.0 to Classify Different Dog Breeds

To do this, we'll be using data from the [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/overview). It consists of a collection of 10,000+ labelled images of 120 different dog breeds.

This kind of problem is called multi-class image classification. It's multi-class because we're trying to classify mutliple different breeds of dog.

We're going to go through the following TensorFlow/Deep Learning workflow:
1. Get data ready (download from Kaggle, store, import).
2. Prepare the data (preprocessing, the 3 sets, X & y).
3. Choose and fit/train a model ([TensorFlow Hub](https://www.tensorflow.org/hub), `tf.keras.applications`, [TensorBoard](https://www.tensorflow.org/tensorboard), [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)).
4. Evaluating a model (making predictions, comparing them with the ground truth labels).
5. Improve the model through experimentation (start with 1000 images, make sure it works, increase the number of images).
6. Save, sharing and reloading your model.

For our machine learning model, we're using MobileNetV2, a lightweight and efficient convolutional neural network architecture, as the base model. It's a pretrained deep learning model available via TensorFlow Hub.

##  Results
1. Achieved 99.85% training accuracy by epoch 15 with very low training loss.
2. On the validation set, the model achieved an accuracy of 84.61%.
3. Submitted to the Kaggle competition:
4. Kaggle Public Leaderboard Log-Loss Score: 0.84608
5. While the training results are strong, the Kaggle score indicates room for improvement in generalization and probability calibration.

