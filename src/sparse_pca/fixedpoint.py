"""Deterministic reference quantization; MP-SPDZ truncation may be probabilistic."""
import numpy as np


def quantize(x, fractional_bits=20):
    return np.rint(np.asarray(x)*2**fractional_bits)/2**fractional_bits


def bounds(x, fractional_bits=20, ring_bits=64, fixedpoint_bits=None):
    c,g = x.shape
    b = float(np.max(np.abs(x), initial=0))
    # Unit vector: |Xv| <= B sqrt(G). Conservative raw fixed-point products,
    # including products accumulated before truncation and squared norms.
    intermediate = max(1., b*g**0.5, 2*c*b*b*g**0.5)
    raw = max(1., g*intermediate**2, c*b*b*g, intermediate*b*c)
    required = int(np.ceil(np.log2(raw)))+2*fractional_bits+3
    required_fixed=int(np.ceil(np.log2(max(1.,g*intermediate**2,c*b))))+fractional_bits+3
    return {"input_abs_bound": b, "intermediate_abs_bound": intermediate,
            "required_ring_bits": required, "ring_bits": ring_bits,
            "required_fixedpoint_bits":required_fixed,
            "fixedpoint_bits":fixedpoint_bits,
            "safe": required <= ring_bits and (fixedpoint_bits is None or required_fixed<=fixedpoint_bits)}


def operator(x,v,fractional_bits=20):
    q=lambda a: quantize(a,fractional_bits)
    x,v=q(x),q(v)
    mu=q(q(x.sum(axis=0))/len(x))
    u=q(x@v)
    return q(q(x.T@u)-len(x)*q(mu*q(mu@v)))
