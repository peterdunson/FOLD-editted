#include <RcppArmadillo.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

arma::mat makeSigma(arma::vec S_D, arma::vec S_LT){
  int d = S_D.n_rows;
  arma::mat E = arma::zeros(d,d);
  E(trimatl_ind(size(E),-1)) = S_LT;
  arma::mat Sigma = arma::diagmat(S_D) + E + E.t();
  return(Sigma);
}

arma::cube makeSigmacube(arma::mat Sig_diag, arma::mat Sig_LT){
  int n = Sig_diag.n_rows;
  int d = Sig_diag.n_cols;
  arma::cube Sigma = arma::zeros(d,d,n);
  for (int i=0; i<n; i++){
    Sigma.slice(i) = makeSigma(Sig_diag.row(i).t(),Sig_LT.row(i).t());
  }
  return(Sigma);
}

arma::vec makeSigmaDet(arma::cube Sigma, int n){
  arma::vec dt = arma::zeros(n);
  for (int i=0; i<n; i++){
    dt(i) = exp(0.25 * arma::log_det_sympd(Sigma.slice(i)));
  }
  return (dt);
}

// Helper function for Jensen-Shannon Divergence between two probability vectors
double jsd(const arma::vec& P, const arma::vec& Q) {
  arma::vec M = 0.5 * (P + Q);
  double KLD_PM = arma::accu(P % (arma::log(P + 1e-10) - arma::log(M + 1e-10)));
  double KLD_QM = arma::accu(Q % (arma::log(Q + 1e-10) - arma::log(M + 1e-10)));
  return 0.5 * (KLD_PM + KLD_QM);
}

// [[Rcpp::export]]
arma::mat mnorm_D_arma(arma::mat mu, arma::mat Sig_diag, arma::mat Sig_LT){
  int n = mu.n_rows;
  arma::mat D = arma::zeros(n,n);
  arma::cube Sigma = makeSigmacube(Sig_diag, Sig_LT);

  for (int i = 0; i < (n - 1); i++) {
    for (int j = i + 1; j < n; j++) {
      arma::vec P = mu.row(i).t();
      arma::vec Q = mu.row(j).t();
      D(i, j) = D(j, i) = sqrt(jsd(P, Q)); // Calculate JSD and assign to distance matrix
    }
  }
  return(D);
}

// [[Rcpp::export]]
arma::mat unorm_D_arma(arma::vec mu, arma::vec sigma){
  int n = mu.n_rows;
  arma::mat D = arma::zeros(n,n);

  for (int i = 0; i < (n - 1); i++) {
    for (int j = i + 1; j < n; j++) {
      arma::vec P = {mu(i), sigma(i)};
      arma::vec Q = {mu(j), sigma(j)};
      D(i, j) = D(j, i) = sqrt(jsd(P, Q));
    }
  }
  return(D);
}

// [[Rcpp::export]]
arma::vec makeJensenShannonAvg(arma::cube theta, int d, int n) {
  int S = theta.n_slices;
  arma::vec jsd_avg = arma::zeros(n*(n-1)/2);
  arma::mat theta_s = arma::zeros(n, 2*d + (d * (d-1)/2));
  arma::mat mu = theta_s.cols(0, d-1);
  arma::mat Sig_diag = theta_s.cols(d, 2*d-1);
  arma::mat Sig_LT = theta_s.cols(2*d, 2*d + (d * (d-1)/2) - 1);
  arma::mat D = arma::zeros(n,n);
  arma::uvec Du_ind = arma::trimatu_ind(size(D),1);

  for (int s = 0; s < S; s++) {
    theta_s = theta.slice(s);
    mu = theta_s.cols(0, d-1);
    Sig_diag = theta_s.cols(d, 2*d-1);
    Sig_LT = theta_s.cols(2*d, 2*d + (d * (d-1)/2) - 1);
    D = mnorm_D_arma(mu, Sig_diag, Sig_LT);
    jsd_avg = D(Du_ind) + jsd_avg;
  }
  jsd_avg = (1.0 / S) * jsd_avg;
  return(jsd_avg);
}

// [[Rcpp::export]]
arma::vec makeuJensenShannonAvg(arma::cube theta, int n) {
  int S = theta.n_slices;
  arma::vec jsd_avg = arma::zeros(n*(n-1)/2);
  arma::mat theta_s = arma::zeros(n, 2);
  arma::vec mu = theta_s.col(0);
  arma::vec sigma = arma::sqrt(theta_s.col(1));
  arma::mat D = arma::zeros(n,n);
  arma::uvec Du_ind = arma::trimatu_ind(size(D),1);

  for (int s = 0; s < S; s++) {
    theta_s = theta.slice(s);
    mu = theta_s.col(0);
    sigma = arma::sqrt(theta_s.col(1));
    D = unorm_D_arma(mu, sigma);
    jsd_avg = D(Du_ind) + jsd_avg;
  }
  jsd_avg = (1.0 / S) * jsd_avg;
  return(jsd_avg);
}
