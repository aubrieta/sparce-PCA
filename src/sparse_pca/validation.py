import numpy as np
from .data import load
from .fixedpoint import bounds, operator as fixed_operator
from .oblivious import matvec
from .reference import dense_operator, distributed_operator, sparse_operator, initial_vector, oracle, power_iteration
from .metrics import relative_error, pc_metrics


def private_operator(x,v,capacity):
    entries=[(int(i),int(j),float(x[i,j])) for i,j in zip(*np.nonzero(x))]
    if len(entries)>capacity: raise ValueError("capacity exceeded")
    entries += [(0,0,0.)]*(capacity-len(entries))
    choose=lambda bit,y,n:y if bit else n
    u=matvec(entries,v,len(x),choose)
    w=np.asarray(matvec([(j,i,z) for i,j,z in entries],u,x.shape[1],choose))
    mu=x.mean(axis=0)
    return w-len(x)*mu*(mu@v)


def validate_data(directory):
    parts,manifest=load(directory); x=np.vstack(parts); c=manifest['config']
    v=initial_vector(x.shape[1],c['seed']); z=x-x.mean(axis=0)
    expected=z.T@(z@v)
    checks={}
    for name,actual in [('dense',dense_operator(x,v)),('sparse',sparse_operator(x,v)),
                        ('distributed',distributed_operator(parts,v))]:
        checks[name]=bool(np.allclose(actual,expected,rtol=1e-10,atol=1e-10))
    # Shared network exercised with clear arithmetic: correctness only, no MPC claim.
    if sum(np.count_nonzero(p) for p in parts)<=256 and x.shape[1]<=64:
        actual=private_operator(x,v,sum(np.count_nonzero(p) for p in parts))
        checks['private_network_cleartext']=bool(np.allclose(actual,expected,rtol=1e-10,atol=1e-10))
    scale=c.get('input_scale',1.)
    fp_error=relative_error(fixed_operator(x*scale,v,c.get('fractional_bits',20))/scale**2,expected)
    checks['quantized_operator']=fp_error<1e-3
    pv,pl=power_iteration(x,c.get('iterations',20),c['seed'])
    rv,rl=oracle(x)
    return {'status':'passed' if all(checks.values()) else 'failed','checks':checks,
            'quantized_operator_relative_error':fp_error,'pc1_metrics':pc_metrics(x,pv,pl,rv,rl),
            'numeric_bounds':bounds(x,c.get('fractional_bits',20),c.get('ring_bits',128),c.get('fixedpoint_bits',40)),
            'mpc_status':'awaiting_execution'}
