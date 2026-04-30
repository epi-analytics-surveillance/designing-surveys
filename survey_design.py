"""Functions for simulating and evaluating the survey designs.
"""

import model
import arviz
import cmdstanpy
from cmdstanpy import CmdStanModel
import numpy as np
import random


def make_surveys(m, spacing_prevs, sample_size_prevs, spacing_seros, sample_size_seros, width=1, duration=1):
    """Make a list of survey objects with given design parameters.
    """

    all_observers = []

    for spacing_prev, sample_size_prev, spacing_sero, sample_size_sero in \
            zip(spacing_prevs, sample_size_prevs, spacing_seros, sample_size_seros):
        
        infection_test = model.Test()
        infection_test.positive_states = [model.InfectionStatus.INFECTED]

        sero_test = model.Test()
        sero_test.positive_states = [model.InfectionStatus.INFECTED, model.InfectionStatus.RECOVERED]

        if width == 1:
            prev_survey_starts = [10 + spacing_prev * i for i in range(365)]
            prev_survey_starts = [i for i in prev_survey_starts if i <= 365]
            infection_survey = model.Survey(m, infection_test, prev_survey_starts, sample_size_prev, duration)

            sero_survey_starts = [10 + spacing_sero * i for i in range(365)]
            sero_survey_starts = [i for i in sero_survey_starts if i <= 365]
            sero_survey = model.Survey(m, sero_test, sero_survey_starts, sample_size_sero, duration)

            all_observers += [infection_survey, sero_survey]

        elif width > 1:
            full_infection_surveys = []
            for survey_start in [10 + spacing_prev * i for i in range(365)]:
                if survey_start < 365:
                    for j in range(width):
                        full_infection_surveys.append(survey_start + j)
            
            full_sero_surveys = []
            for survey_start in [10 + spacing_sero * i for i in range(365)]:
                if survey_start < 365:
                    for j in range(width):
                        full_sero_surveys.append(survey_start + j)

            infection_survey = model.Survey(m, infection_test, full_infection_surveys, sample_size_prev // width, 1)
            sero_survey = model.Survey(m, sero_test, full_sero_surveys, sample_size_sero // width, 1)
            all_observers += [infection_survey, sero_survey]

    return all_observers


def run_simulations(m, observers, transmission_rate, f_infectious, f_recovered):
    """Run one epidemic simulation with all given surveys.
    """

    m.transmission_rate = transmission_rate

    m.f_infectious = f_infectious
    m.f_recovered = f_recovered

    m.add_observers(observers)

    df = m.simulate(365)

    return df


def create_init(incidence, a=None):

    incidence = np.asarray(incidence)
    cases = np.random.binomial(incidence, 0.5)
    if a is None:
        a = random.uniform(0.2, 0.8)
        incidence = cases / a

    log_incidence = np.log(incidence + 1)

    eps = np.diff(log_incidence)
    eps[0] = 0.0
    eps[incidence[:-1] < 10] = 0.0  # Small incidence -> large log-change. Such values are causing overflow error when trying to evalute log-prob.

    rw_init = np.std(np.abs(eps))
    eps_std = eps / rw_init

    inits = {
        'eps_std': eps_std,
        'rw': rw_init,
        'init_incidence': np.log(50),
        }
    
    return inits


def run_inference(df, observers, inference_model, delay_inf, delay_recov, filename, init_mcmc=1000, rw=1):
    """Run MCMC inference for true infections from survey data.
    """

    fits = []

    for infection_survey, sero_survey in zip(observers[::2], observers[1::2]):

        stan_data = {
            'T': len(df['transmissions']),
            'N_delay_inf': len(delay_inf),
            'delay_inf': delay_inf,
            'num_prev': len(infection_survey.num_tested),
            'prev_survey_positives': infection_survey.num_positive,
            'prev_survey_tested': infection_survey.num_tested,
            'prev_times': infection_survey.times,
            'N_delay_sero': len(delay_recov),
            'delay_sero': delay_recov,
            'num_sero': len(sero_survey.num_tested),
            'sero_survey_positives': sero_survey.num_positive,
            'sero_survey_tested': sero_survey.num_tested,
            'sero_times': sero_survey.times,
            'N': 1000000,
            'rw_order': rw,
            'sigma_sero': 1,
            'sigma_prev': 1,
        }

        cmdstanpy.write_stan_json('stan_data_{}.json'.format(filename), stan_data)

        best = None
        best_score = -np.inf
        for _ in range(25):
            init = create_init(df['transmissions'])
            try:
                p = inference_model.log_prob(init, stan_data)['lp__'][0]
            except RuntimeError:
                p = -np.inf
            if p > best_score:
                second_best = best
                best = init
                best_score = p
        if best is None or second_best is None:
            best = create_init(df['transmissions'], a=0.5)
            second_best = create_init(df['transmissions'], a=0.5)

        converged = False
        num_failed_attempts = 0
        num_mcmc = init_mcmc

        while not converged:

            fit = inference_model.sample(data='stan_data_{}.json'.format(filename),
                iter_warmup=num_mcmc // 2,
                iter_sampling=num_mcmc,
                seed=123,
                chains=2,
                adapt_delta=0.99,
                max_treedepth=20,
                output_dir='output_{}'.format(filename),
                inits=[best, second_best],
            )

            results = arviz.from_cmdstanpy(fit)
            rhat = arviz.rhat(results)
            max_rhat = max(rhat['log_incidence'])

            if max_rhat > 1.05:
                num_mcmc *= 2
                num_failed_attempts += 1

            else:
                converged = True

            if num_failed_attempts > 6:
                fit = None
                converged = True

        fits.append(fit)

    return fits


def evaluate_fits(df, fits):
    """Compute CRPS comparing true infections and inferred infections.
    """

    true_infections = np.asarray(df['transmissions'])[7:]
    log_true_infections = np.log(true_infections + 0.01)
    
    scores = []
    expanded_scores = []
    scores_energy = []

    coverages = []
    widths = []
    
    for fit in fits:
        inferred_mcmc_samples = np.log(fit.incidence[:, 7:])
        score, expanded_score = model.crps(inferred_mcmc_samples, log_true_infections)
        scores.append(score)
        expanded_scores.append(expanded_score)

        inferred_mcmc_samples = fit.incidence[:, 7:]
        scores_energy.append(model.energy_score(inferred_mcmc_samples, true_infections))

        coverages.append(model.coverage(inferred_mcmc_samples, true_infections))
        widths.append(model.width(inferred_mcmc_samples, true_infections))

    return scores, expanded_scores, scores_energy, coverages, widths