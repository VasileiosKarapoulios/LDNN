## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, eval = FALSE)

## -----------------------------------------------------------------------------
#  library(LDNN)
#  set.seed(12345)
#  #Train dummy data
#  X1 <- matrix(runif(500*20), nrow=500, ncol=20)
#  X2 <- matrix(runif(500*24), nrow=500, ncol=24)
#  X3 <- matrix(runif(500*24), nrow=500, ncol=24)
#  X4 <- matrix(runif(500*24), nrow=500, ncol=24)
#  X5 <- matrix(runif(500*16), nrow=500, ncol=16)
#  X6 <- matrix(runif(500*16), nrow=500, ncol=16)
#  X7 <- matrix(runif(500*16), nrow=500, ncol=16)
#  X8 <- matrix(runif(500*16), nrow=500, ncol=16)
#  X9 <- matrix(runif(500*16), nrow=500, ncol=16)
#  X10 <- matrix(runif(500*15), nrow=500, ncol=15)
#  Xif <- matrix(runif(500*232), nrow=500, ncol=232)
#  y <- matrix(runif(500), nrow=500, ncol=1)
#  #Test dummy data
#  X1_test <- matrix(runif(500*20), nrow=500, ncol=20)
#  X2_test <- matrix(runif(500*24), nrow=500, ncol=24)
#  X3_test <- matrix(runif(500*24), nrow=500, ncol=24)
#  X4_test <- matrix(runif(500*24), nrow=500, ncol=24)
#  X5_test <- matrix(runif(500*16), nrow=500, ncol=16)
#  X6_test <- matrix(runif(500*16), nrow=500, ncol=16)
#  X7_test <- matrix(runif(500*16), nrow=500, ncol=16)
#  X8_test <- matrix(runif(500*16), nrow=500, ncol=16)
#  X9_test <- matrix(runif(500*16), nrow=500, ncol=16)
#  X10_test <- matrix(runif(500*15), nrow=500, ncol=15)
#  Xif_test <- matrix(runif(500*232), nrow=500, ncol=232)
#  y_test <- matrix(runif(500), nrow=500, ncol=1)
#  #Create the model
#  model = create_model(rnn_inputs = c(20,24,24,24,16,16,16,16,16,15),
#                recurrent_droppout = c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1),
#                inputs = 232,
#                layer_dropout = c(0.1,0.1),
#                n_nodes_hidden_layers = c(1024,1024),
#                loss_function = 'mean_squared_error',
#                opt = 'adam',
#                metric = 'mean_absolute_error')
#  #Fit the model
#  fitted_model = fit_model(model = model,ver = 0, n_epoch = 1,bsize = 32,X1 = X1, X2 = X2, X3 = X3,X4 = X4,X5 = X5,X6 = X6,X7 = X7,X8 = X8,X9 = X9,X10 = X10,Xif = Xif,y = y)
#  #Evaluate the model on test data
#  evaluate_model(model = fitted_model,X1_test = X1_test, X2_test = X2_test, X3_test = X3_test,X4_test = X4_test,X5_test = X5_test,X6_test = X6_test,X7_test = X7_test,X8_test = X8_test,X9_test = X9_test,X10_test = X10_test,Xif_test = Xif_test,y_test = y_test,bsize = 32)

