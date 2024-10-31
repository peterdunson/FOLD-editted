#' Expected Jensen-Shannon Divergence matrix
#'
#' Computes the posterior expected Jensen-Shannon Divergence (JSD) matrix for all localized densities.
#' @param theta List of samples from the localized atoms in a GMM. Each item in the list is an MCMC sample of the localized atoms for the data.
#' @param p Dimension of the data.
#' @param n Sample size, number of objects to cluster.
#' @importFrom Rcpp sourceCpp
#' @return An \code{n}x\code{n} symmetric matrix of the pairwise expected JSD for all objects.
#' @export
#' @useDynLib foldcluster
comp_delta <- function(theta, p, n) {
  # Reshape theta to a 3D array if it's currently a list of matrices
  if (is.list(theta)) {
    theta <- array(unlist(theta), dim = c(n, 2 * p + choose(p, 2), length(theta)))
  }
  
  # Initialize Delta matrix
  Delta <- matrix(0, nrow = n, ncol = n)
  
  # Compute JSD values based on dimension p
  if (p == 1) {
    Delta[upper.tri(Delta)] <- makeuJensenShannonAvg(theta = theta, n = n)
  } else {
    Delta[upper.tri(Delta)] <- makeJensenShannonAvg(theta = theta, d = p, n = n)
  }
  
  # Make Delta symmetric
  Delta <- Delta + t(Delta)
  
  return(Delta)
}

#' FOLD risk function
#'
#' Computes the risk of any clustering according to the FOLD loss.
#' @param c A vector of cluster labels with length \code{n}.
#' @param Delta An \code{n}x\code{n} symmetric matrix of pairwise expected Jensen-Shannon Divergences.
#' @param omega FOLD separation parameter, positive.
#' @importFrom Rcpp sourceCpp
#' @return The risk of the supplied clustering.
#' @export
#' @useDynLib foldcluster
fold_risk <- function(c, Delta, omega) {
  return(risk_cpp(c, Delta, omega))
}

#' Minimize FOLD risk function
#'
#' Minimizes the FOLD risk function over a matrix of clusterings.
#' @param c A matrix, each row is a vector of cluster labels.
#' @param Delta An \code{n}x\code{n} symmetric matrix of pairwise expected Jensen-Shannon Divergences.
#' @param omega FOLD separation parameter, positive.
#' @importFrom Rcpp sourceCpp
#' @return A vector, the clustering which minimizes the FOLD risk.
#' @export
#' @useDynLib foldcluster
min_risk <- function(c, Delta, omega) {
  return(as.vector(minimize_risk_cpp(c, Delta, omega)))
}
