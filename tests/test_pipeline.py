import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sparse_pca.data import prepare,load
from sparse_pca.export import export
from sparse_pca.fixedpoint import bounds,operator
from sparse_pca.reference import dense_operator,initial_vector,power_iteration
from sparse_pca.runner import parse_log,verify_job,write_json
from sparse_pca.cli import bundle


class PipelineTests(unittest.TestCase):
    def config(self):
        return {'cells':6,'genes':3,'density':.3,'seed':7,'value_bound':2,'capacity_per_hospital':6,
                'fractional_bits':20,'fixedpoint_bits':40,'ring_bits':128,'iterations':20,'zero_threshold':.001}

    def test_prepare_export_and_numeric_model(self):
        with tempfile.TemporaryDirectory() as d:
            manifest=prepare(self.config(),d); parts,_=load(d)
            x=np.vstack(parts); v=initial_vector(3,7)
            np.testing.assert_allclose(operator(x,v),dense_operator(x,v),rtol=1e-3,atol=1e-5)
            for variant in ['dense','sparse_private','sparse_public']:
                public,numeric=export(parts,self.config(),Path(d)/variant,variant)
                self.assertTrue(numeric['safe']); self.assertEqual(public['cells'],6)
                for i,p in enumerate(parts):
                    values=(Path(d)/variant/f'Input-P{i}-0').read_text().split()
                    self.assertEqual(len(values),3+(p.size if variant=='dense' else 18))
            self.assertEqual(manifest['nnz'],5)

    def test_capacity_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.config(); cfg['capacity_per_hospital']=0
            with self.assertRaises(ValueError): prepare(cfg,d)

    def test_overflow_rejection(self):
        self.assertFalse(bounds(np.ones((6,3))*1e6,20,64,40)['safe'])

    def test_parse_log(self):
        data=parse_log('PC1 0.1 -0.2 0.3\nEIGENVALUE 1.2\nTime3 = 0.004 seconds\nData sent = 0.01 MB\nGlobal data sent = 0.03 MB\n12345  maximum resident set size\n')
        self.assertEqual(data['vector'],[.1,-.2,.3]); self.assertEqual(data['timers']['3'],.004)
        self.assertEqual(data['sent_mb'],.01); self.assertEqual(data['peak_rss_bytes'],12345)

    def test_verification_and_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); prepare(self.config(),root); parts,_=load(root)
            job={'status':'compiled','variant':'dense','task':'pc1',
                 'public':{'seed':7,'iterations':20}}
            write_json(root/'job.json',job)
            self.assertEqual(verify_job(root)['status'],'awaiting_execution')
            v,lam=power_iteration(np.vstack(parts),20,7)
            write_json(root/'execution.json',{'status':'executed','wall_seconds':1.})
            (root/'party-0.log').write_text('PC1 '+' '.join(map(str,v))+'\nEIGENVALUE '+str(lam)+'\n')
            for i in [1,2]: (root/f'party-{i}.log').write_text('')
            self.assertEqual(verify_job(root)['status'],'passed')
            # This is a fabricated parser fixture, never an MPC execution claim.
            (root/'P0.key').write_text('secret')
            import tarfile
            with tarfile.open(bundle(root)) as archive:
                names=archive.getnames()
                self.assertIn('verification.json',names)
                self.assertNotIn('input.npz',names); self.assertNotIn('P0.key',names)

    def test_malformed_operator_is_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); prepare(self.config(),root)
            write_json(root/'job.json',{'variant':'dense','task':'operator','public':{'seed':7}})
            write_json(root/'execution.json',{'status':'executed'})
            (root/'party-0.log').write_text('VALIDATION_OPERATOR 1\n')
            for i in [1,2]: (root/f'party-{i}.log').write_text('')
            self.assertEqual(verify_job(root)['status'],'failed')
