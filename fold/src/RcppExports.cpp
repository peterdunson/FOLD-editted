// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// mnorm_D_arma
arma::mat mnorm_D_arma(arma::mat mu, arma::mat Sig_diag, arma::mat Sig_LT);
RcppExport SEXP _fold_mnorm_D_arma(SEXP muSEXP, SEXP Sig_diagSEXP, SEXP Sig_LTSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Sig_diag(Sig_diagSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Sig_LT(Sig_LTSEXP);
    rcpp_result_gen = Rcpp::wrap(mnorm_D_arma(mu, Sig_diag, Sig_LT));
    return rcpp_result_gen;
END_RCPP
}
// unorm_D_arma
arma::mat unorm_D_arma(arma::vec mu, arma::vec sigma);
RcppExport SEXP _fold_unorm_D_arma(SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(unorm_D_arma(mu, sigma));
    return rcpp_result_gen;
END_RCPP
}
// makeHellingerAvg
arma::vec makeHellingerAvg(arma::cube theta, int d, int n);
RcppExport SEXP _fold_makeHellingerAvg(SEXP thetaSEXP, SEXP dSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(makeHellingerAvg(theta, d, n));
    return rcpp_result_gen;
END_RCPP
}
// makeuHellingerAvg
arma::vec makeuHellingerAvg(arma::cube theta, int n);
RcppExport SEXP _fold_makeuHellingerAvg(SEXP thetaSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(makeuHellingerAvg(theta, n));
    return rcpp_result_gen;
END_RCPP
}
// risk_cpp
double risk_cpp(arma::vec c, arma::mat Delta, double omega);
RcppExport SEXP _fold_risk_cpp(SEXP cSEXP, SEXP DeltaSEXP, SEXP omegaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type c(cSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Delta(DeltaSEXP);
    Rcpp::traits::input_parameter< double >::type omega(omegaSEXP);
    rcpp_result_gen = Rcpp::wrap(risk_cpp(c, Delta, omega));
    return rcpp_result_gen;
END_RCPP
}
// rand_index
double rand_index(arma::vec c1, arma::vec c2);
RcppExport SEXP _fold_rand_index(SEXP c1SEXP, SEXP c2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c2(c2SEXP);
    rcpp_result_gen = Rcpp::wrap(rand_index(c1, c2));
    return rcpp_result_gen;
END_RCPP
}
// minimize_risk_cpp
arma::vec minimize_risk_cpp(arma::mat c, arma::mat Delta, double omega);
RcppExport SEXP _fold_minimize_risk_cpp(SEXP cSEXP, SEXP DeltaSEXP, SEXP omegaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type c(cSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Delta(DeltaSEXP);
    Rcpp::traits::input_parameter< double >::type omega(omegaSEXP);
    rcpp_result_gen = Rcpp::wrap(minimize_risk_cpp(c, Delta, omega));
    return rcpp_result_gen;
END_RCPP
}
// ldmvnorm_arma
double ldmvnorm_arma(arma::vec y, arma::vec mu, arma::mat Sigma, int d);
RcppExport SEXP _fold_ldmvnorm_arma(SEXP ySEXP, SEXP muSEXP, SEXP SigmaSEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(ldmvnorm_arma(y, mu, Sigma, d));
    return rcpp_result_gen;
END_RCPP
}
// ldunorm_arma
double ldunorm_arma(double y, double mu, double sigma_sq);
RcppExport SEXP _fold_ldunorm_arma(SEXP ySEXP, SEXP muSEXP, SEXP sigma_sqSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_sq(sigma_sqSEXP);
    rcpp_result_gen = Rcpp::wrap(ldunorm_arma(y, mu, sigma_sq));
    return rcpp_result_gen;
END_RCPP
}
// maketau
arma::mat maketau(arma::vec Pi, arma::mat y, arma::mat mu, arma::cube Sigma);
RcppExport SEXP _fold_maketau(SEXP PiSEXP, SEXP ySEXP, SEXP muSEXP, SEXP SigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type Pi(PiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type Sigma(SigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(maketau(Pi, y, mu, Sigma));
    return rcpp_result_gen;
END_RCPP
}
// makeutau
arma::mat makeutau(arma::vec Pi, arma::vec y, arma::vec mu, arma::vec Sigma);
RcppExport SEXP _fold_makeutau(SEXP PiSEXP, SEXP ySEXP, SEXP muSEXP, SEXP SigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type Pi(PiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type Sigma(SigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(makeutau(Pi, y, mu, Sigma));
    return rcpp_result_gen;
END_RCPP
}
// lmaketau
arma::mat lmaketau(arma::vec Pi, arma::mat y, arma::mat mu, arma::mat Sigma);
RcppExport SEXP _fold_lmaketau(SEXP PiSEXP, SEXP ySEXP, SEXP muSEXP, SEXP SigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type Pi(PiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Sigma(SigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(lmaketau(Pi, y, mu, Sigma));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_fold_mnorm_D_arma", (DL_FUNC) &_fold_mnorm_D_arma, 3},
    {"_fold_unorm_D_arma", (DL_FUNC) &_fold_unorm_D_arma, 2},
    {"_fold_makeHellingerAvg", (DL_FUNC) &_fold_makeHellingerAvg, 3},
    {"_fold_makeuHellingerAvg", (DL_FUNC) &_fold_makeuHellingerAvg, 2},
    {"_fold_risk_cpp", (DL_FUNC) &_fold_risk_cpp, 3},
    {"_fold_rand_index", (DL_FUNC) &_fold_rand_index, 2},
    {"_fold_minimize_risk_cpp", (DL_FUNC) &_fold_minimize_risk_cpp, 3},
    {"_fold_ldmvnorm_arma", (DL_FUNC) &_fold_ldmvnorm_arma, 4},
    {"_fold_ldunorm_arma", (DL_FUNC) &_fold_ldunorm_arma, 3},
    {"_fold_maketau", (DL_FUNC) &_fold_maketau, 4},
    {"_fold_makeutau", (DL_FUNC) &_fold_makeutau, 4},
    {"_fold_lmaketau", (DL_FUNC) &_fold_lmaketau, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_fold(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}