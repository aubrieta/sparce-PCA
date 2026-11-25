import shutil
import tempfile
import unittest
from pathlib import Path
from sparse_pca.runner import prepare_certificates


@unittest.skipUnless(shutil.which('openssl'),'OpenSSL required for certificate integration')
class CertificateTests(unittest.TestCase):
    def test_generate_verify_reuse_and_repair(self):
        # Exercises real cryptographic tooling without opening sockets.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); runtime=root/'runtime'; log=root/'ssl.log'
            prepare_certificates(runtime,log)
            data=runtime/'Player-Data'
            first=[(data/f'P{i}.pem').read_bytes() for i in range(3)]
            prepare_certificates(runtime,log)
            self.assertEqual(first,[(data/f'P{i}.pem').read_bytes() for i in range(3)])
            # A wrong identity in an otherwise readable certificate is rejected
            # and repaired, rather than bypassing certificate verification.
            (data/'P1.pem').write_bytes(first[0])
            prepare_certificates(runtime,log)
            self.assertNotEqual((data/'P1.pem').read_bytes(),first[0])
            self.assertIn('P2.pem: OK',log.read_text())
