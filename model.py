"""Functionality of epidemics and surveillance observations.
"""

import bisect
from collections import defaultdict
import copy
from enum import Enum
import random
import numpy as np
import pandas
import scipy.stats


class InfectionStatus(Enum):
    """The various possible states of a persons infection with the disease.
    """
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2


def sample_discrete(p):
    """Generate one sample from a discrete distribution.

    Parameters
    ----------
    p : Probabilities of each index.

    Returns
    -------
    int
        Index that was sampled.
    """
    cumulative_distribution = np.cumsum(p)
    return bisect.bisect(cumulative_distribution, np.random.random() *
                         cumulative_distribution[-1])


def crps(mcmc_samples, true_value):
    """Compute continuous ranked probability score (CRPS).

    Parameters
    ----------
    mcmc_samples : np.ndarray
        MCMC samples of the posterior. Can be either 1d or 2d. If 2d, the
        dimensions should take the form: (mcmc_iterations, time points).
    true_value : float or np.ndarray
        True value or true values.

    Returns
    -------
    float
        CRPS
    """
    mcmc_samples = np.asarray(mcmc_samples)

    if mcmc_samples.ndim == 2:
        all_crps = []
        for i, true_value_i in enumerate(true_value):
            crps = _crps_1d(mcmc_samples[:, i], true_value_i)
            all_crps.append(crps)
        # crps = np.sum(all_crps)
        crps = np.median(all_crps)

    elif mcmc_samples.ndim == 1:
        crps = _crps_1d(mcmc_samples, true_value)
        all_crps = None

    else:
        raise ValueError('Incorrect number of dimensions for mcmc_samples')

    return crps, all_crps


def _crps_1d(mcmc_samples, true_value):
    posterior = scipy.stats.gaussian_kde(mcmc_samples)

    lower = min(min(mcmc_samples), true_value) - 1
    upper = max(max(mcmc_samples), true_value) + 1

    xgrid = np.linspace(lower, upper, 10000)

    if true_value not in xgrid:
        xgrid = np.asarray(sorted(list(xgrid) + [true_value]))

    dx = xgrid[1] - xgrid[0]
    pdf = posterior.pdf(xgrid)
    cdf = np.cumsum(pdf) * dx

    true_idx = np.where(true_value <= xgrid)[0][0]

    c1 = np.trapezoid(cdf[:true_idx] ** 2, xgrid[:true_idx])
    c2 = np.trapezoid((cdf[true_idx:] - 1) ** 2, xgrid[true_idx:])

    return c1 + c2


def _crps_1d_old(mcmc_samples, true_value):
    posterior = scipy.stats.ecdf(mcmc_samples)
    quantiles = posterior.cdf.quantiles
    probabilities = posterior.cdf.probabilities

    if max(mcmc_samples) < true_value:
        crps = np.diff(quantiles) @ probabilities[:-1] ** 2

    elif min(mcmc_samples) > true_value:
        crps = np.diff(quantiles) @ (probabilities[:-1] - 1) ** 2

    else:
        quantiles_below_true = quantiles[np.where(true_value > quantiles)]
        probabilities_below_true = probabilities[np.where(true_value > quantiles)]

        quantiles_above_true = quantiles[np.where(true_value <= quantiles)]
        probabilities_above_true = probabilities[np.where(true_value <= quantiles)]

        if len(quantiles_below_true) > 0:
            quantiles_below_true = np.append(quantiles_below_true, true_value)

        if len(probabilities_below_true) > 0:
            probabilities_above_true = np.insert(probabilities_above_true, 0, probabilities_below_true[-1])
            quantiles_above_true = np.insert(quantiles_above_true, 0, true_value)

        crps = np.diff(quantiles_below_true) @ probabilities_below_true ** 2 \
            + np.diff(quantiles_above_true) @ (probabilities_above_true[:-1] - 1) ** 2

    return crps


class Person:
    """One person in the agent based model.

    Attributes
    ----------
    infection_status : InfectionStatus
        The person's current infection status
    number_of_times_infected : int
        The number of times this person has become infected
    """
    def __init__(self, model):
        """
        Parameters
        ----------
        model : Model
            Model object that this agent will be part of
        """
        self.infection_status = InfectionStatus.SUSCEPTIBLE
        self.model = model
        self.next_status_update = None
        self.number_of_times_infected = 0

    def update_status(self, t):
        """Update to the next infection status within InfectionStatus.
        """
        # Update myself
        old_status = self.infection_status
        next_status = \
            InfectionStatus((old_status.value + 1) % len(InfectionStatus))
        self.infection_status = next_status

        # Update model
        self.model.persons[old_status].remove_person(self)
        self.model.persons[next_status].add_person(self)

        # If entering the Infected or Recovered statuses, decide how long
        # to spend in that status before the next transition. If needed,
        # record the next transition and the number of times infected.
        if next_status == InfectionStatus.INFECTED:
            delay = sample_discrete(self.model.f_infectious)
            self.next_status_update = t + delay
            self.model.transitions[self.next_status_update].append(self)
            self.number_of_times_infected += 1

        elif next_status == InfectionStatus.RECOVERED:
            delay = sample_discrete(self.model.f_recovered)
            self.next_status_update = t + delay
            self.model.transitions[self.next_status_update].append(self)

        elif next_status == InfectionStatus.SUSCEPTIBLE:
            self.next_status_update = None


class PersonCollection:
    """A collection of persons.

    It supports adding people, removing people, and randomly sampling people
    faster than just a list, by simultaneously saving the index of each
    person in a dictionary.

    See https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice  # noqa
    """
    def __init__(self):
        """Initialize an empty collection.
        """
        self.persons = []
        self.persons_map = dict()

    def add_person(self, person):
        """Add a person to this collection.

        Parameters
        ----------
        person : Person
            Person to be added
        """
        if person not in self.persons_map:
            self.persons.append(person)
            self.persons_map[person] = len(self.persons) - 1

    def remove_person(self, person):
        """Remove a person from this collection.

        Parameters
        ----------
        person : Person
            Person to be removed
        """
        map_location = self.persons_map.pop(person)
        last_person = self.persons.pop()

        # If the person requested to be removed happened to be the last, no
        # further action is necessay. Otherwise, now slot in the last person
        # wherever the person was removed.
        if map_location == len(self.persons):
            return
        self.persons[map_location] = last_person
        self.persons_map[last_person] = map_location

    def random_people(self, num):
        """Select persons, at random, without replacement from the collection.

        Parameters
        ----------
        num : int
            Number to randomly sample

        Returns
        -------
        list
            List of persons who happened to be chosen
        """
        return random.sample(self.persons, num)

    def __iter__(self):
        return iter(self.persons)

    def __len__(self):
        return len(self.persons)

    def __contains__(self, person):
        return person in self.persons_map

    def __getitem__(self, i):
        return self.persons[i]


class ModelStep:
    """One step in the agent based model.

    Attributes
    ----------
    model : AgentModel
        Model to which this step is attached.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, time):
        """Run this step at the indicated time point.

        Parameters
        ----------
        time : int
            Time step
        """
        raise NotImplementedError
    

class InfectionProgressionStep(ModelStep):
    """Transition persons from I to R, or from R to S.
    """

    def __call__(self, time):
        for person in copy.copy(self.model.transitions[time]):
            self.model.transitions[time].remove(person)
            person.update_status(time)


class TransmissionStep(ModelStep):
    """Infects persons from S to I, based on transmission from other
    infectious.
    """
    def __init__(self, model):
        super().__init__(model)
        self.num_infected = 0
    
    def _transmission_rate(self, num_susceptible, time):
        r = self.model.transmission_rate(time) * num_susceptible \
            / self.model.N
        return r
    
    def _compute_number_to_infect(self, rate):
        return np.random.poisson(rate)

    def __call__(self, time):
        infected_this_time_step = self.model.persons[InfectionStatus.INFECTED]
        num_susceptible_this_time_step = \
            len(self.model.persons[InfectionStatus.SUSCEPTIBLE])
        
        self.num_infected = 0

        for _ in copy.copy(infected_this_time_step.persons):
            rate_to_infect = \
                self._transmission_rate(num_susceptible_this_time_step, time)
            
            if rate_to_infect <= 0:
                continue

            num_to_infect = self._compute_number_to_infect(rate_to_infect)

            if num_to_infect > num_susceptible_this_time_step:
                num_to_infect = num_susceptible_this_time_step

            persons_to_infect = \
                self.model.persons[InfectionStatus.SUSCEPTIBLE]\
                    .random_people(num_to_infect)
            
            for person in persons_to_infect:
                person.update_status(time)

            self.num_infected += num_to_infect
            num_susceptible_this_time_step -= num_to_infect


class Test:
    """Test of someone for being positive or negative to given infection statuses.

    Attributes
    ----------
    sensitivity : float
        Probability of testing positive, when applied to a truly positive
    specificity : float
        Probability of testing negative, when applied to a truly negative
    positive_statuses : list of simsurveillance.InfectionStatus
        Which states count as truly positive for this test.
    """
    def __init__(self, sensitivity=1.0, specificity=1.0):
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.positive_states = [InfectionStatus.INFECTED]

    def __call__(self, person):
        """Obtain the test result of testing a person.

        Parameters
        ----------
        person : Person
            Person to be tested.
        time : int
            Time when this test is being conducted
        """
        if person.infection_status in self.positive_states:
            if random.random() < self.sensitivity:
                return True
            else:
                return False

        else:
            if random.random() < self.specificity:
                return False
            else:
                return True
            

class Observer:
    """Observation process of an epidemic.

    An observer is called at each simulation time step of the agent based
    model. It can extract exact information or simulate some testing process
    applied to a subset of the population.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : simsurveillance.AgentModel
            Agent based model
        """
        self.model = model

    def __call__(self, time):
        raise NotImplementedError
    

class Survey(Observer):

    def __init__(self, model, test, start_dates, sample_size, duration):
        super().__init__(model)

        self.test = test
        self.start_dates = start_dates
        self.sample_size = sample_size
        self.duration = duration

        self.times = []
        self.num_tested = []
        self.num_positive = []

        self.continue_dates = []
        self.chunk_progress = 1

    def __call__(self, time):
        if time in self.start_dates:

            to_be_tested = self.model.all_persons.random_people(self.sample_size)

            if self.duration > 1:
                self.continue_dates = []
                for i in range(self.duration - 1):
                    self.continue_dates.append(time + i + 1)

                self.chunks = []

                n = self.sample_size // self.duration + 1
                for i in range(0, len(to_be_tested), n):
                    self.chunks += [to_be_tested[i:i + n]]

                to_be_tested = self.chunks[0]
                self.chunk_progress = 1

            num_positive = 0
            for p in to_be_tested:
                result = self.test(p)
                if result:
                    num_positive += 1

            if self.duration > 1:
                time += self.duration // 2

            if self.duration == 1:
                self.times.append(time)
            else:
                self.times.append(time + self.duration // 2)
            
            self.num_tested.append(len(to_be_tested))
            self.num_positive.append(num_positive)

        elif time in self.continue_dates:
            to_be_tested = self.chunks[self.chunk_progress]
            self.chunk_progress += 1

            num_positive = 0
            for p in to_be_tested:
                result = self.test(p)
                if result:
                    num_positive += 1

            self.num_tested[-1] += len(to_be_tested)
            self.num_positive[-1] += num_positive


class AgentModel:

    def __init__(self, N, num_init_infected):
        self.N = N
        self.transitions = defaultdict(list)
        self.new_infectees = defaultdict(list)
        
        self.persons = {status: PersonCollection() for status in InfectionStatus}
        self.all_persons = PersonCollection()

        self.f_infectious = [0, 0, 0.1, 0.5, 0.4]
        self.f_recovered = [0] * 25 + [1]
        
        # Initialize all agents as susceptible
        for _ in range(N):
            a = Person(self)
            self.persons[InfectionStatus.SUSCEPTIBLE].add_person(a)
            self.all_persons.add_person(a)

        # Set initial infected agents
        to_infect = self.persons[InfectionStatus.SUSCEPTIBLE].random_people(num_init_infected)
        for a in to_infect:
            a.update_status(0)

        self.transmission_rate = lambda x: 1.0
        self.case_ascertainment = lambda x: 1.0

        self.transmission_step = TransmissionStep(self)
        self.infection_progression_step = InfectionProgressionStep(self)

        self.observers = []

    def add_observers(self, observers):
        self.observers = observers

    def simulate(self, T):
        output = defaultdict(list)

        for t in range(1, T+1):

            output['time'].append(t)
            for status in InfectionStatus:
                output[status].append(len(self.persons[status]))
            output['transmissions'].append(
                self.transmission_step.num_infected)

            # Simulation steps --------------------------
            self.transmission_step(t)
            self.infection_progression_step(t)
            # -------------------------------------------

            # Observation steps --------------------------
            for observer in self.observers:
                observer(t)
            # -------------------------------------------

        return pandas.DataFrame(output)


def energy_score(mcmc_samples, true_value):

    mcmc_samples = np.asarray(mcmc_samples)
    true_value = np.asarray(true_value)

    n_mcmc = mcmc_samples.shape[0]
    n_samples = 5000

    ds1 = []
    ds2 = []
    beta = 1

    for _ in range(n_samples):
        X = mcmc_samples[np.random.randint(n_mcmc, size=1), :]
        Xp = mcmc_samples[np.random.randint(n_mcmc, size=1), :]

        ds1.append(np.linalg.norm(np.abs(X - true_value)) ** beta)
        ds2.append(np.linalg.norm(np.abs(X - Xp)) ** beta)

    score = np.mean(ds1) - 0.5 * np.mean(ds2)
    return score


def coverage(mcmc_samples, true_value):
    
    mcmc_samples = np.asarray(mcmc_samples)
    true_value = np.asarray(true_value)

    lower = np.percentile(mcmc_samples, 2.5, axis=0)
    upper = np.percentile(mcmc_samples, 97.5, axis=0)

    covered = (lower < true_value) & (true_value < upper)
    prop_covered = sum(covered) / len(covered)

    return prop_covered


def width(mcmc_samples, true_value):
    
    mcmc_samples = np.asarray(mcmc_samples)
    true_value = np.asarray(true_value)

    lower = np.percentile(mcmc_samples, 2.5, axis=0)
    upper = np.percentile(mcmc_samples, 97.5, axis=0)

    width = upper - lower

    return np.mean(width)