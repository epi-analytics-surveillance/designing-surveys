data {
  int T;  // Number of days

  int N_delay_inf;  // Length of infection positivity profile
  vector[N_delay_inf] delay_inf;  // Infection positivity profile

  int num_prev;  // Number of infection prevalence survey observations
  array[num_prev] int prev_survey_positives;  // Numbers positive
  array[num_prev] int prev_survey_tested;  // Total numbers tested
  array[num_prev] int prev_times;  // Times at which survey took place
  
  int N_delay_sero;  // Length of seropositivity profile
  vector[N_delay_sero] delay_sero;  // Seropositivity profile

  int num_sero;  // Number of seroprevalence survey observations
  array[num_sero] int sero_survey_positives;  // Numbers positive
  array[num_sero] int sero_survey_tested;  // Total numbers tested
  array[num_sero] int sero_times;  // Times at which survey took place

  int N;  // Total underlying population size
  
  real sigma_sero;
  real sigma_prev;
  
  int rw_order;
}


parameters {
  real<lower=0> rw;
  real<lower=0, upper=10> init_incidence;
  vector[T-rw_order] eps_std;
}


transformed parameters {
  vector[T] log_incidence;
  vector[T-rw_order] eps;
  
  if(rw_order == 1) {
    log_incidence[1] = init_incidence;
    eps = rw * eps_std;
    log_incidence[2:] = log_incidence[1] + cumulative_sum(eps);
  } else if (rw_order == 2) {
    log_incidence[1] = init_incidence;
    log_incidence[2] = init_incidence;
    eps = rw * eps_std;
    for(j in 1:T-2){
      log_incidence[2 + j] = eps[j] + 2 * log_incidence[1 + j] - log_incidence[j];
    }
  }

  vector[T] incidence = exp(log_incidence);

  array[num_prev] real prevalence;
  for(j in 1:num_prev) {
    int time = prev_times[j];
    int K = min(N_delay_inf, time - 1);
    real s2 = dot_product(delay_inf[1:K], reverse(incidence[1:time])[1:K]);
    prevalence[j] = s2;
  }
  
  array[num_sero] real seroprevalence;
  for(j in 1:num_sero) {
    int time = sero_times[j];
    int K = min(N_delay_sero, time - 1);
    real s2 = dot_product(delay_sero[1:K], reverse(incidence[1:time])[1:K]);
    seroprevalence[j] = s2;
  }
}


model {
  init_incidence ~ uniform(0, 10);
  eps_std ~ std_normal();

  rw ~ normal(0, 0.05) T[0.001, ];

  for(j in 1:num_prev) {
    target += binomial_lpmf(prev_survey_positives[j] | prev_survey_tested[j], prevalence[j] / N);
    
    // real mean = N * (1.0 + prev_survey_positives[j]) / (2.0 + prev_survey_tested[j]);
    // real vari = N * (1.0 + prev_survey_tested[j] - prev_survey_positives[j]) * (1.0 + prev_survey_positives[j]) / pow(2.0 + prev_survey_tested[j], 2.0) * (2.0 + prev_survey_tested[j] + N) / (3.0 + prev_survey_tested[j]);
    // target += normal_lpdf(prevalence[j] | mean, sigma_prev * sqrt(vari));
  }
  
  for(j in 1:num_sero) {
    target += binomial_lpmf(sero_survey_positives[j] | sero_survey_tested[j], seroprevalence[j] / N);
  
    // real mean = N * (1.0 + sero_survey_positives[j]) / (2.0 + sero_survey_tested[j]);
    // real vari = N * (1.0 + sero_survey_tested[j] - sero_survey_positives[j]) * (1.0 + sero_survey_positives[j]) / pow(2.0 + sero_survey_tested[j], 2.0) * (2.0 + sero_survey_tested[j] + N) / (3.0 + sero_survey_tested[j]);
    // target += normal_lpdf(seroprevalence[j] | mean, sigma_sero * sqrt(vari));
  }

}

