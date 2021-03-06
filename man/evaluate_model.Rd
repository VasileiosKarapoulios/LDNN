% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/evaluate_model.R
\name{evaluate_model}
\alias{evaluate_model}
\title{Evaluate the pre-defined Neural Network for Longitudinal Data}
\usage{
evaluate_model(
  model,
  X1_test,
  X2_test,
  X3_test,
  X4_test,
  X5_test,
  X6_test,
  X7_test,
  X8_test,
  X9_test,
  X10_test,
  Xif_test,
  y_test,
  bsize
)
}
\arguments{
\item{model}{The fitted model object produced by create_model().}

\item{X1_test}{Features as inputs of 1st LSTM.}

\item{X2_test}{Features as inputs of 2nd LSTM.}

\item{X3_test}{Features as inputs of 3rd LSTM.}

\item{X4_test}{Features as inputs of 4th LSTM.}

\item{X5_test}{Features as inputs of 5th LSTM.}

\item{X6_test}{Features as inputs of 6th LSTM.}

\item{X7_test}{Features as inputs of 7th LSTM.}

\item{X8_test}{Features as inputs of 8th LSTM.}

\item{X9_test}{Features as inputs of 9th LSTM.}

\item{X10_test}{Features as inputs of 10th LSTM.}

\item{Xif_test}{The features to be concatenated with the outputs of the LSTMs.}

\item{y_test}{The target variable.}

\item{bsize}{The batch size.}
}
\value{
The evaluation results.
}
\description{
Evaluate the fitted Neural Network model (Keras).
}
\examples{
X1_test <- matrix(runif(500*20), nrow=500, ncol=20)
X2_test <- matrix(runif(500*24), nrow=500, ncol=24)
X3_test <- matrix(runif(500*24), nrow=500, ncol=24)
X4_test <- matrix(runif(500*24), nrow=500, ncol=24)
X5_test <- matrix(runif(500*16), nrow=500, ncol=16)
X6_test <- matrix(runif(500*16), nrow=500, ncol=16)
X7_test <- matrix(runif(500*16), nrow=500, ncol=16)
X8_test <- matrix(runif(500*16), nrow=500, ncol=16)
X9_test <- matrix(runif(500*16), nrow=500, ncol=16)
X10_test <- matrix(runif(500*15), nrow=500, ncol=15)
Xif_test <- matrix(runif(500*232), nrow=500, ncol=232)
y_test <- matrix(runif(500), nrow=500, ncol=1)
\dontrun{
evaluate_model(fitted_model,X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,
X7_test,X8_test,X9_test,X10_test,Xif_test,y_test,32)
}
# The functions require to have python installed
# As well as tensorflow, keras and reticulate package.
}
