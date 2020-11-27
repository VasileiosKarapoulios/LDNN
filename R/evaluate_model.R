#' Evaluate the pre-defined Neural Network for Longitudinal Data 
#' 
#' @param model The fitted model object produced by create_model().
#' @param X1_test Features as inputs of 1st LSTM.
#' @param X2_test Features as inputs of 2nd LSTM.
#' @param X3_test Features as inputs of 3rd LSTM.
#' @param X4_test Features as inputs of 4th LSTM.
#' @param X5_test Features as inputs of 5th LSTM.
#' @param X6_test Features as inputs of 6th LSTM.
#' @param X7_test Features as inputs of 7th LSTM.
#' @param X8_test Features as inputs of 8th LSTM.
#' @param X9_test Features as inputs of 9th LSTM.
#' @param X10_test Features as inputs of 10th LSTM.
#' @param Xif_test The features to be concatenated with the outputs of the LSTMs.
#' @param y_test The target variable.
#' @param bsize The batch size.
#' @return The evaluation results.
#' @description Evaluate the fitted Neural Network model (Keras).
#' @examples
#' X1_test <- matrix(runif(500*20), nrow=500, ncol=20)
#' X2_test <- matrix(runif(500*24), nrow=500, ncol=24)
#' X3_test <- matrix(runif(500*24), nrow=500, ncol=24)
#' X4_test <- matrix(runif(500*24), nrow=500, ncol=24)
#' X5_test <- matrix(runif(500*16), nrow=500, ncol=16)
#' X6_test <- matrix(runif(500*16), nrow=500, ncol=16)
#' X7_test <- matrix(runif(500*16), nrow=500, ncol=16)
#' X8_test <- matrix(runif(500*16), nrow=500, ncol=16)
#' X9_test <- matrix(runif(500*16), nrow=500, ncol=16)
#' X10_test <- matrix(runif(500*15), nrow=500, ncol=15)
#' Xif_test <- matrix(runif(500*232), nrow=500, ncol=232)
#' y_test <- matrix(runif(500), nrow=500, ncol=1)
#' \dontrun{
#' evaluate_model(fitted_model,X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,
#' X7_test,X8_test,X9_test,X10_test,Xif_test,y_test,32)
#' }
#' # The functions require to have python installed
#' # As well as tensorflow, keras and reticulate package.
#' @import keras
#' @export evaluate_model
evaluate_model <- function(model, X1_test, X2_test, X3_test, X4_test, X5_test, X6_test, X7_test, X8_test, X9_test, X10_test, Xif_test, y_test, bsize){
  res = model %>% evaluate(list(inp1 = X1_test,
                                inp2 = X2_test,
                                inp3 = X3_test,
                                inp4 = X4_test,
                                inp5 = X5_test,
                                inp6 = X6_test,
                                inp7 = X7_test,
                                inp8 = X8_test,
                                inp9 = X9_test,
                                inp10 = X10_test,
                                inpif = Xif_test), y_test, batch_size = bsize)
  return(res)
}