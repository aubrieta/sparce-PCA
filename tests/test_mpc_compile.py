"""Opt-in compiler integration; does not start any party processes."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from sparse_pca.data import prepare
from sparse_pca.runner import compile_suite,ROOT


@unittest.skipUnless(os.environ.get('SPCA_TEST_COMPILE')=='1','set SPCA_TEST_COMPILE=1 to compile all six programs')
class CompilerTests(unittest.TestCase):
    def test_compile_all_variants(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            prepare(json.loads((ROOT/'configs/smoke.json').read_text()),root/'data')
            jobs=compile_suite(root/'data',root,os.environ.get('MP_SPDZ_HOME','/tmp/MP-SPDZ'))
            self.assertEqual(len(jobs),6)
            for name in jobs:
                self.assertEqual(json.loads((root/name/'job.json').read_text())['status'],'compiled')
