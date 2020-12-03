#' Pre-defined Neural Network for Longitudinal Data 
#' 
#' @param rnn_inputs The number of inputs (integers) per each LSTM (vector of length 10).
#' @param recurrent_droppout The dropout to be applied in the LSTMs (between 0 and 1).
#' @param inputs The number of inputs (integer) to be concatenated with the output of the LSTMs.
#' @param layer_dropout The dropout to be applied between the hidden layers (between 0 and 1).
#' @param n_nodes_hidden_layers The number of nodes in the hidden layers (2 in total).
#' @param loss_function The loss function to be used.
#' @param opt The optimizer to be used.
#' @param metric The metric to be used.
#' @return The model object built in Keras.
#' @description Create the Neural Network model (Keras).
#' @examples
#' inp = c(20,24,24,24,16,16,16,16,16,15)
#' rec_drop = rep(0.1,10)
#' l_drop = c(0.1,0.1)
#' \dontrun{
#' create_model(inp,rec_drop,232,c(0.1,0.1),l_drop,'mean_squared_error','adam','mean_absolute_error')
#' }
#' # The functions require to have python installed
#' # As well as tensorflow, keras and reticulate package.
#' @import keras
#' @export create_model
create_model<-function(rnn_inputs, recurrent_droppout, inputs, layer_dropout, n_nodes_hidden_layers, loss_function, opt, metric){
  inp1 <- layer_input(shape = c(rnn_inputs[1],1), name="inp1")
  inp2 <- layer_input(shape = c(rnn_inputs[2],1), name="inp2")
  inp3 <- layer_input(shape = c(rnn_inputs[3],1), name="inp3")
  inp4 <- layer_input(shape = c(rnn_inputs[4],1), name="inp4")
  inp5 <- layer_input(shape = c(rnn_inputs[5],1), name="inp5")
  inp6 <- layer_input(shape = c(rnn_inputs[6],1), name="inp6")
  inp7 <- layer_input(shape = c(rnn_inputs[7],1), name="inp7")
  inp8 <- layer_input(shape = c(rnn_inputs[8],1), name="inp8")
  inp9 <- layer_input(shape = c(rnn_inputs[9],1), name="inp9")
  inp10 <- layer_input(shape = c(rnn_inputs[10],1), name="inp10")
  
  rnn1 <- inp1 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[1], kernel_initializer="he_normal") 
  
  rnn2 <- inp2 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[2], kernel_initializer="he_normal") 
  
  rnn3 <- inp3 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[3], kernel_initializer="he_normal") 
  
  rnn4 <- inp4 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[4], kernel_initializer="he_normal") 
  
  rnn5 <- inp5 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[5], kernel_initializer="he_normal") 
  
  rnn6 <- inp6 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[6], kernel_initializer="he_normal") 
  
  rnn7 <- inp7 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[7], kernel_initializer="he_normal") 
  
  rnn8 <- inp8 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[8], kernel_initializer="he_normal") 
  
  rnn9 <- inp9 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[9], kernel_initializer="he_normal") 
  
  rnn10 <- inp10 %>%
    layer_lstm(units = 1, recurrent_dropout = recurrent_droppout[10], kernel_initializer="he_normal") 
  
  inpif <- layer_input(shape = c(inputs), name = "inpif")
  
  out <- layer_concatenate(c(rnn1, rnn2, rnn3, rnn4, rnn5, rnn6, rnn7, rnn8, rnn9, rnn10, inpif), name = "concat") %>%
    layer_dropout(layer_dropout[1]) %>%
    layer_dense(units = n_nodes_hidden_layers[1], activation = "relu", kernel_initializer="he_normal") %>%
    layer_dropout(layer_dropout[2]) %>%
    layer_dense(units = n_nodes_hidden_layers[2], activation = "relu", kernel_initializer="he_normal") %>%
    layer_dense(units = 1, name="output") # output
  
  model_constructed <- keras_model(
    inputs = c(inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,inpif),
    outputs = c(out)
  )
  
  model_constructed %>% compile(loss = loss_function, 
                                optimizer = opt,
                                metrics = metric)
  
  return(model_constructed)
}