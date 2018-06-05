# C. Muhs
# 6/2/18
# Cascadia Rconf
# Workshop 1 - Intermediate ML

# Following stuff from https://tensorflow.rstudio.com/keras/


require("devtools")
devtools::install_github("rstudio/keras")
library(keras)
install_keras()

# MNIST data
mnist <- dataset_mnist()
# Define training and testing data sets
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Define model

model <- keras_model_sequential() 
model %>% 
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax') # 10 digits to predict

summary(model)

# compile model
model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)

# Training and evaluation

history <- model %>% fit(
    x_train, y_train, 
    epochs = 30,  # retrain 30 times. Takes some time
    batch_size = 128, 
    validation_split = 0.2
)

plot(history)

# Evaluate the model’s performance on the test data:
model %>% evaluate(x_test, y_test)

# Generate predictions on new data:
model %>% predict_classes(x_test)
