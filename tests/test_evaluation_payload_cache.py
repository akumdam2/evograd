from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from evograd.evaluation.payload_cache import case_payload_cache
from evograd.verification.payloads import Corpus
from tests.verification.test_tier2_inputs import write_corpus


class PayloadCacheTests(unittest.TestCase):
    def test_one_storage_read_across_agents_with_private_alias_preserving_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            first = write_corpus(Path(directory), [(303, 0)])
            second = Corpus.open(first.manifest_path)
            record = first.payloads()[0]
            original = Corpus._read_storage
            reads = []

            def read(corpus, key, device):
                reads.append(key)
                return original(corpus, key, device)

            with patch.object(Corpus, "_read_storage", read), case_payload_cache([first, second]):
                a = first.load(record, device="cpu")
                saved = a["a"].clone()
                a["a"].add_(3)
                b = second.load(record, device="cpu")
                self.assertTrue(torch.equal(b["a"], saved))
                self.assertEqual(len(reads), len(set(reads)))
                self.assertEqual(b["a"].untyped_storage().data_ptr(), b["b"].untyped_storage().data_ptr())
            self.assertNotIn("_read_storage", first.__dict__)
            self.assertNotIn("_read_storage", second.__dict__)
