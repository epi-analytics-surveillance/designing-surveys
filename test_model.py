import unittest
import unittest.mock
from unittest.mock import patch
import numpy as np
import scipy.stats
import scipy.integrate
import model


class TestInfectionStatus(unittest.TestCase):

    def test_infection_status(self):
        susceptible = model.InfectionStatus.SUSCEPTIBLE
        self.assertEqual(susceptible.name, 'SUSCEPTIBLE')
        self.assertEqual(susceptible.value, 0)

        infected = model.InfectionStatus.INFECTED
        self.assertEqual(infected.name, 'INFECTED')
        self.assertEqual(infected.value, 1)

        recovered = model.InfectionStatus.RECOVERED
        self.assertEqual(recovered.name, 'RECOVERED')
        self.assertEqual(recovered.value, 2)


class TestSampleDiscrete(unittest.TestCase):

    def test_sample_discrete(self):
        p = [0, 0, 1]
        x = model.sample_discrete(p)
        self.assertEqual(x, 2)

        p = [0, 0.25, 0.75, 0]
        xs = []
        for _ in range(100):
            xs.append(model.sample_discrete(p))
        self.assertSetEqual({1, 2}, set(xs))

        p = [0.1, 0.2, 0.3, 0.4]
        xs = []
        for _ in range(1000):
            xs.append(model.sample_discrete(p))
        self.assertGreater(xs.count(1), xs.count(0))
        self.assertGreater(xs.count(2), xs.count(1))
        self.assertGreater(xs.count(3), xs.count(2))


class TestCRPS(unittest.TestCase):

    def test_crps(self):
        test_true_value = -0.5
        test_mean = 1.0
        test_std = 5.0
        def f(x):
            if x < test_true_value:
                return scipy.stats.norm.cdf(x, test_mean, test_std) ** 2
            else:
                return (scipy.stats.norm.cdf(x, test_mean, test_std) - 1) ** 2
            
        expected_crps = scipy.integrate.quad(f, -50, 50)[0]
        samples = scipy.stats.norm.rvs(test_mean, test_std, size=10000)
        computed_crps = model.crps(samples, test_true_value)
        self.assertAlmostEqual(expected_crps, computed_crps, delta=0.05 * expected_crps)

        test_true_value = 0.24
        test_mean = 0.25
        test_std = 0.1
        def f(x):
            if x < test_true_value:
                return scipy.stats.norm.cdf(x, test_mean, test_std) ** 2
            else:
                return (scipy.stats.norm.cdf(x, test_mean, test_std) - 1) ** 2
            
        expected_crps = scipy.integrate.quad(f, -10, 10)[0]
        samples = scipy.stats.norm.rvs(test_mean, test_std, size=10000)
        computed_crps = model.crps(samples, test_true_value)
        self.assertAlmostEqual(expected_crps, computed_crps, delta=0.05 * expected_crps)
    
        test_true_value = -1.5
        test_mean = 1.0
        test_std = 2.5
        def f(x):
            if x < test_true_value:
                return scipy.stats.norm.cdf(x, test_mean, test_std) ** 2
            else:
                return (scipy.stats.norm.cdf(x, test_mean, test_std) - 1) ** 2
            
        expected_crps = scipy.integrate.quad(f, -10, 10)[0]
        samples = scipy.stats.norm.rvs(test_mean, test_std, size=10000)
        computed_crps = model.crps(samples, test_true_value)
        self.assertAlmostEqual(expected_crps, computed_crps, delta=0.05 * expected_crps)


class TestAgent(unittest.TestCase):

    @patch('model.AgentModel')
    def test_init(self, mock_model):
        person = model.Person(mock_model)
        self.assertEqual(person.infection_status, model.InfectionStatus.SUSCEPTIBLE)
        self.assertEqual(person.model, mock_model)

    @patch('model.AgentModel')
    def test_update_status(self, mock_model):
        person = model.Person(mock_model)

        self.assertEqual(person.infection_status, model.InfectionStatus.SUSCEPTIBLE)
        person.update_status(10)
        self.assertEqual(person.infection_status, model.InfectionStatus.INFECTED)
        person.update_status(10)
        self.assertEqual(person.infection_status, model.InfectionStatus.RECOVERED)


class TestPersonCollection(unittest.TestCase):

    def test_init(self):
        collection = model.PersonCollection()
        self.assertEqual(collection.persons, [])

    @patch('model.Person')
    @patch('model.Person')
    def test_add_person(self, mock_agent1, mock_agent2):
        collection = model.PersonCollection()
        collection.add_person(mock_agent1)
        self.assertIn(mock_agent1, collection.persons)
        self.assertEqual(collection.persons_map[mock_agent1], 0)

        collection.add_person(mock_agent2)
        self.assertIn(mock_agent2, collection.persons)
        self.assertEqual(collection.persons_map[mock_agent1], 0)
        self.assertEqual(collection.persons_map[mock_agent2], 1)

    @patch('model.Person')
    @patch('model.Person')
    def test_remove_person(self, mock_agent1, mock_agent2):
        collection = model.PersonCollection()
        collection.add_person(mock_agent1)
        collection.add_person(mock_agent2)

        collection.remove_person(mock_agent2)
        self.assertNotIn(mock_agent2, collection.persons)
        self.assertEqual(collection.persons_map[mock_agent1], 0)

        collection = model.PersonCollection()
        collection.add_person(mock_agent1)
        collection.add_person(mock_agent2)

        collection.remove_person(mock_agent1)
        self.assertNotIn(mock_agent1, collection.persons)
        self.assertEqual(collection.persons_map[mock_agent2], 0)

    @patch('model.Person')
    @patch('model.Person')
    @patch('model.Person')
    def test_random_people(self, mock_agent1, mock_agent2, mock_agent3):
        collection = model.PersonCollection()
        collection.add_person(mock_agent1)
        collection.add_person(mock_agent2)
        collection.add_person(mock_agent3)

        sample = collection.random_people(1)
        self.assertEqual(len(sample), 1)
        self.assertTrue(mock_agent1 in sample or mock_agent2 in sample or mock_agent3 in sample)

        sample = collection.random_people(2)
        self.assertEqual(len(sample), 2)
        self.assertTrue(mock_agent1 in sample or mock_agent2 in sample or mock_agent3 in sample)


if __name__ == '__main__':
    unittest.main()

