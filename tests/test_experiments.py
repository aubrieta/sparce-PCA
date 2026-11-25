"""Reporting fixtures are synthetic metadata, not MPC execution evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sparse_pca.experiments import compile_sweep,summarize_sweep
from sparse_pca.runner import write_json


class ExperimentTests(unittest.TestCase):
    def test_declared_capacity_and_realized_sparsity(self):
        config={'source':'synthetic','seed':17,'value_bound':4,'warmups':1,'repetitions':5,
                'sweep':{'shapes':[[24,12]],'densities':[.01,.5,1.],
                         'supports':['uniform'],'partitions':[[1/3]*3]}}
        with tempfile.TemporaryDirectory() as d:
            with patch('sparse_pca.experiments.compile_program') as compiler:
                plan=compile_sweep(config,d,'/tmp/MP-SPDZ',variants=['dense'])
            self.assertEqual(compiler.call_count,3)
            self.assertEqual([c['config']['capacity_per_hospital'] for c in plan['cases']],[3,96,96])
            self.assertEqual(plan['cases'][0]['nnz'],3)
            self.assertAlmostEqual(plan['cases'][0]['sparsity'],1-3/288)
            self.assertEqual(plan['cases'][-1]['padded_entries'],288)

    def test_failed_accuracy_never_produces_speedup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); case=root/'case-000'; case.mkdir()
            entry={'directory':'case-000','shape':[24,12],'sparsity':.9,
                   'realized_density':.1,'nnz':29,'padded_entries':87,
                   'jobs':['dense-operator','sparse_private-operator']}
            write_json(root/'experiment.json',{'cases':[entry],'repetitions':2})
            for variant in ['dense','sparse_private']:
                target=case/f'{variant}-operator'; target.mkdir()
                write_json(target/'job.json',{'variant':variant})
            measured=[]
            for variant in ['dense','sparse_private']:
                for i in range(2):
                    measured.append({'variant':variant,'warmup':False,
                        'status':'passed' if variant=='dense' or i==0 else 'failed',
                        'wall_seconds':1.,'kernel_seconds':.5})
            write_json(case/'measurements.json',measured)
            rows=summarize_sweep(root)
            self.assertEqual(rows[0]['status'],'passed')
            self.assertEqual(rows[1]['n_failed'],1)
            self.assertNotIn('wall_seconds_speedup',rows[1])
