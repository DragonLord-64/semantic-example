"""Ensure generated and shipped data support the three-label teaching task."""
import json
from pathlib import Path
import unittest

from generate_dataset import audit, build
from model import LABELS, tokenize


ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_all_labels_are_present_in_each_split(self):
        rows = build()
        audit(rows)
        self.assertEqual(len({row['id'] for row in rows}), len(rows))
        for split in ('train', 'validation', 'test'):
            with self.subTest(split=split):
                self.assertEqual({row['label'] for row in rows if row['split'] == split},
                                 set(LABELS))
        for row in rows:
            self.assertEqual(tokenize(row['sentence'])[row['target_index']], row['target'])

    def test_shipped_splits_match_the_generator(self):
        rows = build()
        for split in ('train', 'validation', 'test'):
            with self.subTest(split=split):
                with (ROOT / f'{split}.jsonl').open() as source:
                    shipped = [json.loads(line) for line in source if line.strip()]
                self.assertEqual(shipped, [row for row in rows if row['split'] == split])


if __name__ == '__main__':
    unittest.main()
