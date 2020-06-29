#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_SPARSE_MATRIX(Q0);
  DATA_SPARSE_MATRIX(I);
  
  
  /* Fishery dependent data */
  DATA_FACTOR(time);      // Length = haulid length
  DATA_FACTOR(gf);        // Length = haulid length
  DATA_FACTOR(rowid);     // Length = haulid length; haulid matching the rows of the dataframe (rowid)
  DATA_VECTOR(response);  // "shorter"
  DATA_MATRIX(X);         // nrow = response length (Both fixed and random)
  DATA_VECTOR(offset);
  //DATA_IVECTOR(fd_indicator); //Length=response length
  //DATA_IVECTOR(support_area); //Length=nlevels gf
  
  /* Random fields */
  PARAMETER_ARRAY(eta_density);
  PARAMETER_VECTOR(eta_nugget); // Not used ! perhaps later.
  
  /* 5 common fixed effects x number of times */
  PARAMETER(logdelta);        // Length=2
  PARAMETER(logscale);         // Dim = c(ntime,2)
  PARAMETER(logsd_nugget);    // Length = ntime. not used
  PARAMETER(time_corr);
  
  
  /* Parameters */
  PARAMETER_VECTOR(beta);         // Fixed effects
  PARAMETER_VECTOR(beta_r);       // Random effects
  PARAMETER_VECTOR(beta_r_logsd);
  DATA_FACTOR(beta_r_fac);
  PARAMETER_VECTOR(logphi); // NB overdispersion
  PARAMETER_VECTOR(alpha);
  
  /* Stuff for prediction */
  DATA_INTEGER(doPredict);  // Flag
  DATA_MATRIX(Xpredict);
  DATA_FACTOR(Apredict);
  
  /* Flat prior for 'robustification' */
  DATA_SCALAR(huge_sd);
  
  /* Distance in KM between grid points  */
  DATA_SCALAR(h);
  
  Type sd_nugget = exp(logsd_nugget);
  Type ans = 0;
  using namespace density;
  
  /* Add *flat* prior to fixed effects */
  ans -= dnorm(beta, Type(0), huge_sd, true).sum();
  
  /* Add random effects beta_r */
  for(int i=0; i<beta_r.size(); i++)
    ans -= dnorm(beta_r(i), Type(0), exp(beta_r_logsd(beta_r_fac(i))), true);
  vector<Type> beta_full(X.cols());
  beta_full << beta, beta_r;
  if(offset.size() == 0) {
    offset.resize(X.rows());
    offset.setZero();
  }
  
  
  /* Optional: Add static field */
  PARAMETER_VECTOR(eta_static);
  PARAMETER_VECTOR(logdelta_static);
  PARAMETER_VECTOR(logscale_static);
  if(eta_static.size() > 0) {
    /* Scale parameters for fields */
    Type scale = exp(logscale_static[0]);
    /* GMRF: Sparse precision matrix */
    Eigen::SparseMatrix<Type> Q = Q0 + exp(logdelta_static[0]) * I;
    GMRF_t<Type> nldens = GMRF(Q);
    ans += SCALE(nldens, scale)(eta_static);
    int ntimes = NLEVELS(time);
    for(int i=0; i<ntimes; i++) {
      eta_density.col(i) -= eta_static;
    }
    REPORT(ntimes);
  }
  
  /* Time covariance */
  //N01<Type> nldens_time;
  Type phi = time_corr / sqrt(1.0 + time_corr*time_corr);
  AR1_t<N01<Type> > nldens_time = AR1(phi);
  /* Scale parameters for fields */
  Type scale = exp(logscale);
  /* GMRF: Sparse precision matrix */
  Eigen::SparseMatrix<Type> Q = Q0 + exp(logdelta) * I;
  GMRF_t<Type> nldens = GMRF(Q);
  ans += SEPARABLE(SCALE(nldens_time, scale), nldens)(eta_density);
  /* Nugget */
  if(eta_nugget.size() > 0)
    ans -= dnorm(eta_nugget, Type(0), sd_nugget, true).sum();
  
  
  /* Include preferential sampling likelihood */
  DATA_FACTOR(SupportAreaGroup);
  DATA_IARRAY(SupportAreaMatrix);
  for (int group = 0; group < NLEVELS(SupportAreaGroup); group++) {
    vector<int> support_area = SupportAreaMatrix.col(group);
    vector<Type> logsum (NLEVELS(time));
    logsum.setZero(); logsum = log(logsum); // logsum = -INFINITY
    for(int j=0; j<logsum.size();j++){
      for(int i=0; i<NLEVELS(gf); i++){
        if(support_area(i)) logsum(j) = logspace_add(logsum(j), alpha[group]*eta_density(i,j));
      }
    } 
    for(int i=0; i<rowid.size(); i++){
      if(i==0 || (rowid(i)!=rowid(i-1))){
        int pos = gf(i);
        int tim = time(i);
        // density ~ lambda^alpha
        if (SupportAreaGroup(rowid(i)) == group)
          ans -= alpha[group]*eta_density(pos,tim) - logsum(tim);
      }
    }
  }
  
  /* Fishery dep data */
  if(response.size() > 0) {
    vector<Type> log_mu(response.size());
    log_mu.setZero(); log_mu = log(log_mu); // log_mu = -INFINITY
    for(int i=0; i<gf.size(); i++) {
      if(eta_static.size() == 0) {
        log_mu(rowid(i)) = logspace_add(log_mu(rowid(i)),
               eta_density(gf[i], time[i]) );
      } else {
        log_mu(rowid(i)) = logspace_add(log_mu(rowid(i)),
               eta_density(gf[i], time[i])  + eta_static(gf[i]) );
      }
    }
    log_mu = log_mu + X * beta_full + offset;
    /* Use numerical robust negative binomial distribution rather than
    mu = exp(log_mu);
    var = mu + mu*mu / exp(logphi(0));
    ans -= dnbinom2(response, mu, var, true).sum(); */
    vector<Type> log_var_minus_mu = log_mu + log_mu - logphi(0);
    REPORT(log_mu);            // For debugging
    REPORT(log_var_minus_mu);  // For debugging
    ans -= dnbinom_robust(response, log_mu, log_var_minus_mu, true).sum();
  }
  
  if(doPredict){
    array<Type> logindex = eta_density;
    // NOTE: eta_density is the *total* field (including the static if present)
    //if(eta_static.size() > 0) {
    //  for(int j=0; j<logindex.cols(); j++)
    //    logindex.col(j) += eta_static;
    //}
    // FIXME: Add common fixed effects to predictions (incude covariates available at every grid point).
    REPORT(logindex);
    ADREPORT(logindex);
    
    // Spatial decorrelation distance (corr=exp(-1)) - to compute a plot of spatial correlation vs. distance (km)
    Type sigma_square = nldens.variance().mean(); //This represents the first term in eq.3 in Kristensen et al. (2014)
    Type delta = exp(logdelta);
    Type H = h / log(1 + (delta/2) + sqrt(delta + (delta*delta/4)));
    ADREPORT(H);
    
    if(isDouble<Type>::value) {
      Type avgVariance = nldens.variance().mean() * pow(scale, 2.);
      Type SpatialSD = sqrt(avgVariance);
      Type ForecastSD = sqrt(1. - phi*phi) * SpatialSD;
      REPORT(SpatialSD);
      REPORT(ForecastSD);
    }
  }
  
  return ans;
}