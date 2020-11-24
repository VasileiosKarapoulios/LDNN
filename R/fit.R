#' Fit the pre-defined Neural Network for Longitudinal Data 
#' 
#' @param model The model object produced by model().
#' @param ver ver=0 to show nothing, ver=1 to show animated progress bar, ver=2 to just mention the number of epoch during training. 
#' @param n_epoch The number of epochs to train the model.
#' @param bsize The batch size.
#' @param X1 Features as inputs of 1st LSTM.
#' @param X2 Features as inputs of 2nd LSTM.
#' @param X3 Features as inputs of 3rd LSTM.
#' @param X4 Features as inputs of 4th LSTM.
#' @param X5 Features as inputs of 5th LSTM.
#' @param X6 Features as inputs of 6th LSTM.
#' @param X7 Features as inputs of 7th LSTM.
#' @param X8 Features as inputs of 8th LSTM.
#' @param X9 Features as inputs of 9th LSTM.
#' @param X10 Features as inputs of 10th LSTM.
#' @param Xif The features to be concatenated with the outputs of the LSTMs.
#' @param y The target variable.
#' @return The fitted model.
#' @description DESCRIPTION TO BE INCLUDED
#' @examples
#' fit(model = model,
#'     ver = 0,
#'     n_epochs = 200,
#'     bsize = 32,
#'     X1 = X1, 
#'     X2 = X2, 
#'     X3 = X3,
#'     X4 = X4,
#'     X5 = X5,
#'     X6 = X6,
#'     X7 = X7,
#'     X8 = X8,
#'     X9 = X9,
#'     X10 = X10,
#'     Xif = Xif,
#'     y = y)
#' @export fit
fit<-function(model, ver, n_epoch, bsize, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Xif, y){
  set.seed(1234)
  
  checkpoint_path <- "checkpoints/cp.ckpt"
  
  # Create checkpoint callback
  cp_callback <- callback_model_checkpoint(
    monitor = "val_loss", 
    filepath = checkpoint_path,
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0 # not printing
  )
  
  trained_model <- model %>% keras::fit(
    x = list(inp1 = X1,
             inp2 = X2,
             inp3 = X3,
             inp4 = X4,
             inp5 = X5,
             inp6 = X6,
             inp7 = X7,
             inp8 = X8,
             inp9 = X9,
             inp10 = X10,
             inpif = Xif), # sequence we're using for prediction 
  y = y, # sequence we're predicting
  batch_size = bsize, # how many samples to pass to our model at a time
  epochs = n_epoch, # how many times we'll look at the whole dataset
  validation_split = 0.1, # how much data to hold out for testing as we go along
  callbacks = list(cp_callback),  # pass callback to training
  verbose = ver) # not printing during training

  fitted = model %>% load_model_weights_tf(filepath = checkpoint_path)

  return(fitted)
}