import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from sparse_pca.runner import wait_exact


class ProcessClockTests(unittest.TestCase):
    def test_uses_blocking_wait_and_records_each_party(self):
        class Process:
            def wait(self,*args,**kwargs):
                self.args=args; self.kwargs=kwargs
                return 0
        ps=[Process() for _ in range(3)]
        before=time.perf_counter(); records=wait_exact(ps,10)
        self.assertEqual(len(records),3)
        for p,(code,finished) in zip(ps,records):
            self.assertEqual(code,0); self.assertGreaterEqual(finished,before)
            self.assertEqual(p.args,()); self.assertEqual(p.kwargs,{})

    def test_real_local_processes_without_sockets(self):
        ps=[subprocess.Popen([sys.executable,'-c','pass']) for _ in range(3)]
        records=wait_exact(ps,10)
        self.assertTrue(all(code==0 for code,t in records))

    def test_deadline_kills_and_reaps_process(self):
        # Test a genuine blocked process; the production timeout mechanism
        # must not leave waitpid threads preventing shutdown.
        p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
        with self.assertRaises(subprocess.TimeoutExpired): wait_exact([p],.01)
        self.assertIsNotNone(p.poll())
