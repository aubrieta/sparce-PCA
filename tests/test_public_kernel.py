import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sparse_pca.public_kernel import (VARIANTS,fixtures,schedules,arithmetic_bounds,vectors,
    emulate,error,quantize,paired_speedup,operation_counts,audit_api,prepare_fixture,verify)
from sparse_pca.runner import ROOT,write_json


class OptimizedKernelTests(unittest.TestCase):
    def setUp(self): self.cfg=json.loads((ROOT/'configs/public-kernel.json').read_text())

    def test_all_fixtures_original_and_quantized(self):
        for name,parts,seed,main in fixtures(self.cfg,[17,7,29]):
            x=np.vstack(parts); m=np.count_nonzero(x); cap=min(96,max(1,int(m)))
            ss=schedules(parts,cap); arithmetic_bounds(parts,ss,self.cfg)
            vs=vectors(x.shape[1],seed)
            for variant in VARIANTS:
                actual=emulate(parts,ss,vs,self.cfg,variant)
                for matrix,inputs in [(x,vs),(quantize(x*.25)/.25,quantize(vs))]:
                    z=matrix-matrix.mean(0)
                    for a,v in zip(actual,inputs):
                        result=error(a,z.T@(z@v),.001,.0001)
                        if name=='edge-cancellation' and variant in ('sparse_scalar','sparse_grouped'):
                            # Known calibration finding, not an accepted MPC pass.
                            self.assertLess(result['absolute_error'],.002)
                        else:
                            self.assertTrue(result['passed'],(name,variant,result))

    def test_cancellation_failure_is_preserved(self):
        for name,parts,seed,_ in fixtures(self.cfg,[17]):
            if name!='edge-cancellation': continue
            x=np.vstack(parts); ss=schedules(parts,96); vs=vectors(12,seed)
            actual=emulate(parts,ss,vs,self.cfg,'sparse_scalar')
            z=x-x.mean(0)
            failures=[error(a,z.T@(z@v),.001,.0001) for a,v in zip(actual,vs)]
            self.assertTrue(any(not e['passed'] for e in failures))

    def test_schedules_reuse_every_slot_once_per_orientation(self):
        _,parts,_,_=next(fixtures(self.cfg,[17]))
        cap=3; ss=schedules(parts,cap)
        for s in ss:
            self.assertEqual(sorted(i for group in s['row_groups'] for i in group),list(range(cap)))
            self.assertEqual(sorted(i for group in s['columns'] for i in group),list(range(cap)))
        self.assertTrue(any(s['coordinates'].count([0,0])>1 for s in ss))

    def test_width_rejection_even_if_ring_large(self):
        parts=[np.full((8,12),1e4) for _ in range(3)]
        with self.assertRaisesRegex(ValueError,'unsafe accumulation'):
            arithmetic_bounds(parts,schedules(parts,96),self.cfg)

    def test_dot_reductions_and_empty_groups(self):
        for name,parts,seed,main in fixtures(self.cfg,[17]):
            cap=min(96,max(1,int(np.count_nonzero(np.vstack(parts)))))
            cfg={**self.cfg,'schedules':schedules(parts,cap),'genes':12,'cells':24,'capacity':cap}
            scalar=operation_counts(cfg,'sparse_scalar'); grouped=operation_counts(cfg,'sparse_grouped')
            dot=operation_counts(cfg,'sparse_dot')
            self.assertEqual(scalar['matrix_scale_reductions_per_application'],grouped['matrix_scale_reductions_per_application'])
            self.assertLessEqual(dot['matrix_scale_reductions_per_application'],scalar['matrix_scale_reductions_per_application'])

    def test_bootstrap_pairs_not_independent_samples(self):
        a={i:float(i+1) for i in range(10)}; b={i:v/2 for i,v in a.items()}
        stats=paired_speedup(a,b,1000,17)
        self.assertEqual(stats['speedup'],2); self.assertEqual(stats['ci95'],[2,2])
        self.assertTrue(stats['convincing_10_percent_reduction'])
        self.assertEqual(paired_speedup({1:1},{2:1})['n_pairs'],0)

    def test_installed_source_assumptions(self):
        if not Path('/tmp/MP-SPDZ/Compiler/types.py').exists(): self.skipTest('MP-SPDZ not installed')
        with tempfile.TemporaryDirectory() as d: audit_api('/tmp/MP-SPDZ',d)

    def test_output_validation_and_corruption_detection(self):
        # Fabricated log fixture tests parsing/gating, not secure execution.
        import shutil
        from sparse_pca.reference import dense_operator
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); _,parts,seed,_=next(fixtures(self.cfg,[17]))
            cfg=prepare_fixture(parts,seed,self.cfg,root)
            cfg.update(workload='validation',vector_count=32)
            shutil.copy2(root/'input.npz',root/'reference.npz')
            write_json(root/'job.json',{'public':cfg})
            write_json(root/'execution.json',{'status':'executed'})
            x=np.vstack(parts); vs=vectors(12,seed)
            ys=[dense_operator(x,v) for v in vs]; check=sum((i+1)*v for i,v in enumerate(ys))
            text='\n'.join('RESULT '+str(i)+' '+' '.join(map(str,y)) for i,y in enumerate(ys))
            text+='\nCHECKSUM '+' '.join(map(str,check))+'\n'
            for i in range(3): (root/f'party-{i}.log').write_text(text if i==0 else '')
            self.assertEqual(verify(root)['status'],'passed')
            original=(root/'verification.json').read_bytes()
            policy={'id':'test-v2','absolute_tolerance':.002,'relative_tolerance':.0001}
            self.assertEqual(verify(root,policy)['status'],'passed')
            self.assertEqual((root/'verification.json').read_bytes(),original)
            self.assertTrue((root/'verification-test-v2.json').exists())
            (root/'party-0.log').write_text(text.replace('RESULT 0 ','RESULT 99 ',1))
            # Out-of-range application indices must fail rather than silently
            # skipping the missing first result.
            result=verify(root)
            self.assertEqual(result['status'],'failed')
