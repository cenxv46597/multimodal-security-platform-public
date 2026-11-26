import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform_app'))
import datasets


class DatasetTests(unittest.TestCase):
    def test_only_two_development_datasets_registered(self):
        self.assertEqual(set(datasets.CONFIG), {'sar81', 'sar332'})
        self.assertEqual(datasets.summary('sar81')['total'], 81)
        self.assertEqual(datasets.summary('sar332')['total'], 332)
        with self.assertRaises(ValueError):
            datasets.summary('heldout243')

    def test_filter_and_pagination(self):
        first = datasets.browse('sar81', 'ship', 1)
        second = datasets.browse('sar81', 'ship', 2)
        self.assertEqual((first['filtered_total'], len(first['items']), len(second['items'])), (15, 12, 3))
        self.assertTrue(all(item['label'] == 'ship' for item in first['items'] + second['items']))
        self.assertTrue(all(set(item) == {'sample_id', 'label', 'label_name', 'image_url'} for item in first['items']))
        with self.assertRaises(ValueError):
            datasets.browse('sar81', 'ship', 3)
        with self.assertRaises(ValueError):
            datasets.browse('sar81', 'unknown', 1)

    def test_registered_image_hash_and_invalid_index(self):
        data, content_type = datasets.image('sar81', 0)
        self.assertGreater(len(data), 0)
        self.assertIn(content_type, {'image/png', 'image/jpeg', 'image/bmp'})
        with self.assertRaises(ValueError):
            datasets.image('sar81', -1)
        with self.assertRaises(ValueError):
            datasets.image('sar81', 81)


if __name__ == '__main__':
    unittest.main()
