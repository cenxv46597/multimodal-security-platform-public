"""Backend selection and compatibility tests for the read-only dashboard."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform_app'))
import catalog


class CatalogTests(unittest.TestCase):
    def test_valid_existing_report_is_not_executable(self):
        config = {'goal': 'robustness', 'method': 'step', 'dataset': 'sar12', 'models': 'local'}
        self.assertEqual([x['id'] for x in catalog.options('methods', config)], ['reeval', 'step'])
        self.assertEqual([x['id'] for x in catalog.options('datasets', config)], ['sar12'])
        self.assertEqual([x['id'] for x in catalog.options('models', config)], ['local'])
        self.assertFalse(catalog.validate(config)['execution_available'])

    def test_only_compatible_local_reevaluation_can_run(self):
        config = {'goal': 'robustness', 'method': 'reeval', 'dataset': 'sar1', 'models': 'internvl2b'}
        self.assertTrue(catalog.validate(config)['execution_available'])
        with self.assertRaises(ValueError):
            catalog.validate({**config, 'dataset': 'sar12'})
        with self.assertRaises(ValueError):
            catalog.validate({**config, 'models': 'local'})

    def test_unimplemented_goal_cannot_use_robustness_method(self):
        config = {'goal': 'injection', 'method': 'step', 'dataset': 'sar12', 'models': 'local'}
        self.assertEqual(catalog.options('methods', config), [])
        with self.assertRaises(ValueError):
            catalog.validate(config)

    def test_wrong_or_missing_selection_is_rejected(self):
        for config in ({}, {'goal': 'robustness', 'method': 'step', 'dataset': 'unknown', 'models': 'local'},
                       {'goal': 'robustness', 'method': 'step', 'dataset': 'sar12', 'models': 'unknown'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                catalog.validate(config)


if __name__ == '__main__':
    unittest.main()
