"""Independent optimized public-support experiment; legacy artifacts are immutable.

Invoke as python -m sparse_pca.public_kernel {compile,run,summarize,bundle}.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
from .data import partition
from .synthetic import generate
from .reference import initial_vector, dense_operator, oracle, power_iteration
from .metrics import pc_metrics
from .runner import ROOT, write_json, environment, run_job, parse_log, compile_program

VARIANTS=('dense','sparse_scalar','sparse_grouped','sparse_dot')


def schedules(parts,capacity):
    result=[]
    for x in parts:
        coordinates=[[int(r),int(c)] for r,c in zip(*np.nonzero(x))]
        if len(coordinates)>capacity: raise ValueError('public capacity exceeded')
        coordinates += [[0,0]]*(capacity-len(coordinates))
        rows=[[i for i,(r,c) in enumerate(coordinates) if r==j] for j in range(len(x))]
        cols=[[i for i,(r,c) in enumerate(coordinates) if c==j] for j in range(x.shape[1])]
        result.append({'rows':len(x),'coordinates':coordinates,'row_groups':rows,'columns':cols})
    return result


def vectors(genes,seed,n=32):
    rng=np.random.default_rng(seed+104729)
    vs=rng.normal(size=(n,genes)); vs/=np.linalg.norm(vs,axis=1)[:,None]
    vs[0]=initial_vector(genes,seed)
    if n>1 and genes>1:
        vs[1]=0; vs[1,:2]=[1/math.sqrt(2),-1/math.sqrt(2)]
    return vs


def quantize(a,f=20): return np.rint(np.asarray(a)*2**f)/2**f


def arithmetic_bounds(parts,schedule,cfg):
    """Bound encoded operands and sums, including widths used by sfix.reduce.

    Installed types.py: sfix.unreduced uses k+f (not 2*k and not log2(d)
    extra bits); unreduced_sfix.reduce_after_mul rounds(k+f,f,signed=True).
    Values below include quantization/error slack. Public unit vectors bound
    each component by 1.01, including approximate normalized PC1 iterates.
    Compiler primitive checks remain mandatory, especially division/sqrt.
    """
    f=cfg['fractional_bits']; k=cfg['fixedpoint_bits']; ring=cfg['ring_bits']
    if (f,k,ring)!=(20,48,128): raise ValueError('matched experiment requires f=20,k=48,ring=128')
    scale=cfg['input_scale']; c=sum(map(len,parts)); g=parts[0].shape[1]
    b=max(1e-6,max(float(np.max(np.abs(p),initial=0))*scale for p in parts))+2**-f
    vinf=1.01; u=g*b*vinf+1e-3
    alpha=g*b*vinf+1e-3
    w=2*c*g*b*b*vinf+1e-2
    Q=2**f
    encoded=lambda bound:math.ceil(bound*Q)+2
    tests=[]
    def group(label,d,a,bound,output=None):
        raw=int(d)*encoded(a)*encoded(bound)
        tests.append({'operation':label,'length':d,'a_max_encoded':encoded(a),
                      'b_max_encoded':encoded(bound),'untruncated_sum_abs_bound':raw,
                      'declared_truncation_bits':k+f,'signed_raw_limit':2**(k+f-1),
                      'safe':raw<2**(k+f-1) and raw<2**(ring-1)})
        if output is not None:
            tests.append({'operation':label+' result','encoded_abs_bound':encoded(output),
                          'declared_result_bits':k,'safe':encoded(output)<2**(k-1)})
    # Check both dense lengths and every padded sparse group; zero padding is
    # conservatively treated as bounded nonzero data when bounding accumulation.
    for p,s in enumerate(schedule):
        for name,groups,a,bound in [('forward',s['row_groups'],b,vinf),
                                    ('transpose',s['columns'],b,u)]:
            group(f'{p} {name} padded',max(map(len,groups),default=0),a,bound)
        group(f'{p} dense forward',g,b,vinf,u)
        group(f'{p} dense transpose',len(parts[p]),b,u)
    group('mean inner product',g,b,vinf,alpha)
    group('mean correction (C*mu)*alpha',1,c*b,alpha,w)
    group('normalization squared norm',g,w,w,g*w*w)
    group('Rayleigh numerator',g,vinf,w,g*vinf*w)
    group('Rayleigh denominator',g,vinf,vinf,c*g*vinf*vinf)
    group('restored outputs',1,w/scale**2,1,w/scale**2)
    checksum_bound=w*sum(range(1,cfg['batch_vectors']+1))/scale**2
    tests.append({'operation':'restored weighted checksum','encoded_abs_bound':encoded(checksum_bound),
                  'safe':encoded(checksum_bound)<2**(k-1)})
    # The active ring truncation implementation calls require_ring_size(k+f).
    # FPDiv minimum internal precision: f_internal=max(f,floor(k/2)+1).
    internal_f=max(f,k//2+1); division_width=k+3*internal_f-f
    requirements={'round_operand_bits':k+f,'truncation_mask_bits':k+f+1,'double_precision_bits':2*k,
                  'division_operand_bits':division_width,'ring_bits':ring}
    safe=all(t['safe'] for t in tests) and max(k+f+1,2*k,division_width)<=ring
    report={'safe':safe,'tests':tests,'protocol_width_requirements':requirements,
            'unit_vector_component_bound':vinf,'normalized_iterate_assumption':
            'PC1 component bound assumes stable normalization; output norm checked after execution',
            'forward_abs_bound':u,'centered_abs_bound':w,
            'truncation_source':'types.py: sfix.unreduced / unreduced_sfix.reduce_after_mul',
            'additional_checks':'actual compiler checks sqrt/division primitive requirements'}
    if not safe: raise ValueError('unsafe accumulation/truncation configuration: '+json.dumps(report))
    return report


def fixtures(cfg,seeds,edges=True):
    for seed in seeds:
        for density in cfg['densities']:
            x=generate(24,12,density,seed,'uniform',4.)
            yield f'seed-{seed}-density-{density:g}',partition(x,[1/3]*3,seed),seed,True
    if edges:
        zero=np.zeros((24,12)); constant=np.full((24,12),4.)
        empty=generate(24,12,.25,17,'uniform',4.); empty[:6]=0; empty[:,8:]=0
        cancellation=np.tile([4.,4.,.4,.4,4.,4.,.4,.4,4.,4.,.4,.4],(24,1))
        skew=generate(24,12,.1,17,'uniform',4.)
        for name,x,cuts in [('zero',zero,[8,16]),('constant',constant,[8,16]),
                            ('empty-rows-columns',empty,[8,16]),('cancellation',cancellation,[8,16]),
                            ('unequal',skew,[3,10])]:
            yield 'edge-'+name,list(np.split(x,cuts)),17,False


def prepare_fixture(parts,seed,cfg,directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    x=np.vstack(parts); m=int(np.count_nonzero(x)); cap=min(96,max(1,m))
    # Unequal-partition correctness cases can have more than 96 entries at a
    # single hospital; fixture used here remains sparse enough for this cap.
    started=time.perf_counter(); schedule=schedules(parts,cap)
    schedule_seconds=time.perf_counter()-started
    bound=arithmetic_bounds(parts,schedule,cfg)
    vs=vectors(x.shape[1],seed,cfg['batch_vectors'])
    quantized=[quantize(p*cfg['input_scale'],cfg['fractional_bits'])/cfg['input_scale'] for p in parts]
    np.savez(directory/'input.npz',**{f'party{i}':p for i,p in enumerate(parts)},vectors=vs,
             **{f'quantized{i}':p for i,p in enumerate(quantized)})
    base={**cfg,'seed':seed,'rows':[len(p) for p in parts],'cells':len(x),'genes':x.shape[1],
          'capacity':cap,'schedules':schedule,'nnz':m,'realized_density':m/x.size,
          'padded_slots':3*cap,'schedule_seconds':schedule_seconds,
          'support_hash':hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest(),
          'input_hash':hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
          'bounds':bound,'support_loading':'public compile-time schedules for all four variants'}
    write_json(directory/'fixture.json',base)
    # Local references must agree before any compilation.
    qx=np.vstack(quantized); qv=quantize(vs,cfg['fractional_bits'])
    checks=[]
    for matrix,inputs in [(x,vs),(qx,qv)]:
        z=matrix-matrix.mean(0)
        checks.extend(bool(np.allclose(z.T@(z@v),dense_operator(matrix,v),atol=1e-9,rtol=1e-9)) for v in inputs)
    if not all(checks): raise ValueError('explicit and implicit references disagree')
    calibration={}
    for variant in VARIANTS:
        actual=emulate(parts,schedule,vs,cfg,variant)
        errs=[error(a,dense_operator(x,v),cfg['absolute_tolerance'],cfg['relative_tolerance']) for a,v in zip(actual,vs)]
        calibration[variant]={'max_absolute_error':max(e['absolute_error'] for e in errs),
                              'passes_requested_tolerance':all(e['passed'] for e in errs),
                              'note':'deterministic quantization model; not MPC execution'}
    write_json(directory/'local-checks.json',{'reference_checks_passed':all(checks),'checks':len(checks),
               'bounds_passed':bound['safe'],'quantization_calibration':calibration,'mpc_status':'awaiting_execution'})
    return base


def operation_counts(cfg,variant):
    schedules_=cfg['schedules']; g=cfg['genes']; c=cfg['cells']; cap=cfg['capacity']
    if variant=='dense': products=2*c*g; reductions=c+3*g; dots=reductions
    else:
        products=6*cap
        groups=[group for s in schedules_ for kind in ['row_groups','columns'] for group in s[kind]]
        reductions=sum(bool(group) for group in groups) if variant=='sparse_dot' else products
        dots=reductions if variant=='sparse_dot' else 0
    return {'matrix_scalar_products_per_application':products,
            'matrix_scale_reductions_per_application':reductions,
            'matrix_dot_calls_per_application':dots,
            'shared_correction_dot_calls':1,'shared_correction_scale_reductions':g+1,
            'classification':'source-derived counts, separately checked against compiler artifacts'}


def emulate(parts,schedule,vs,cfg,variant):
    """Deterministic quantization oracle, not a probabilistic MPC emulator."""
    f=cfg['fractional_bits']; scale=cfg['input_scale']; q=lambda a:quantize(a,f)
    matrices=[q(p*scale) for p in parts]
    means=q(sum(p.sum(0) for p in matrices)/sum(map(len,parts)))
    def multiply(p,vector,transpose=False):
        x=matrices[p]; s=schedule[p]
        if variant=='dense':
            return q(x.T@vector if transpose else x@vector)
        n=int(np.count_nonzero(x)); coords=s['coordinates']
        vals=[x[r,c] if i<n else 0. for i,(r,c) in enumerate(coords)]
        groups=s['columns' if transpose else 'row_groups']
        products=np.array([vals[i]*vector[r if transpose else c] for i,(r,c) in enumerate(coords)])
        if variant!='sparse_dot': products=q(products)
        result=np.array([sum(products[i] for i in group) for group in groups])
        return q(result) if variant=='sparse_dot' else result
    results=[]
    for v in q(vs):
        us=[multiply(p,v) for p in range(3)]
        w=sum(multiply(p,u,True) for p,u in enumerate(us))
        results.append((w-q(len(np.vstack(parts))*means*q(means@v)))/scale**2)
    return np.asarray(results)


def audit_api(mpspdz,directory):
    path=Path(mpspdz)/'Compiler/types.py'; text=path.read_text()
    required=['return cls.unreduced_dot_product(x, y, res_params).reduce_after_mul()',
              'return unreduced_sfix(v, self.k + other.f, other.f)',
              'v = sfix.int_type.round(self.v, self.k, self.m,']
    if not all(s in text for s in required):
        raise RuntimeError('installed fixed-point API differs from audited assumptions; inspect widths before proceeding')
    excerpts=[]; lines=text.splitlines()
    for phrase in required:
        index=next(i for i,line in enumerate(lines) if phrase in line)
        excerpts.append({'line':index+1,'text':'\n'.join(lines[max(0,index-4):index+5])})
    write_json(Path(directory)/'api-audit.json',{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'excerpts':excerpts,'verdict':'integer accumulation precedes one reduction; declared raw width is k+f',
        'environment':environment(mpspdz)})


def compiler_evidence(directory):
    directory=Path(directory); log=(directory/'compile.log').read_text()
    resources={}
    for count,kind in re.findall(r'^\s+(\d+) (integer .+|virtual machine rounds|bit .+)$',log,re.M):
        resources[kind]=resources.get(kind,0)+int(count)
    mnemonics={}; trunc_functions=set()
    for p in (directory/'assembly').glob('*'):
        if not p.is_file(): continue
        for line in p.read_text(errors='replace').splitlines():
            match=re.match(r'^([a-z][a-z0-9_]*)\s',line)
            if match: mnemonics[match[1]]=mnemonics.get(match[1],0)+1
            trunc_functions.update(re.findall(r'TruncPr\([^)]*\)[\w_]*',line))
    return {'compiler_resources':resources,'assembly_instruction_counts':mnemonics,
            'truncation_function_names':sorted(trunc_functions),
            'note':'assembly counts are static instructions, not dynamic lanes or network rounds'}


def compile_job(fixture,variant,workload,mpspdz):
    fixture=Path(fixture); target=fixture/(variant+'-'+workload)
    if target.exists():
        job=json.loads((target/'job.json').read_text())
        if job['status']=='compiled':
            current=hashlib.sha256((ROOT/'mpc/public_kernel.mpc').read_bytes()).hexdigest()
            if job['source_sha256']!=current: raise ValueError('source changed; use a fresh output directory')
            return target.name
        raise ValueError('incomplete job exists; preserve it and use a fresh output directory')
    target.mkdir(); inputs=target/'inputs'; inputs.mkdir()
    base=json.loads((fixture/'fixture.json').read_text())
    cfg={**base,'variant':variant,'workload':workload,
         'vector_count':1 if workload in ('single','pc1') else base['batch_vectors']}
    write_json(inputs/'public.json',cfg)
    with np.load(fixture/'input.npz') as data:
        vs=quantize(data['vectors'][:cfg['vector_count']],cfg['fractional_bits'])
        for party in range(3):
            x=data[f'quantized{party}']*cfg['input_scale']; schedule=cfg['schedules'][party]
            values=x.sum(0).tolist()
            if variant=='dense': values+=x.ravel().tolist()
            else:
                n=int(np.count_nonzero(x))
                values += [float(x[r,col]) if i<n else 0. for i,(r,col) in enumerate(schedule['coordinates'])]
            if party==0: values+=vs.ravel().tolist()
            (inputs/f'Input-P{party}-0').write_text('\n'.join(format(v,'.17g') for v in values)+'\n')
    source=(ROOT/'mpc/public_kernel.mpc').read_text()
    name='pubopt_'+variant+'_'+workload+'_'+uuid.uuid4().hex[:10]
    root=Path(mpspdz).resolve(); (root/'Programs/Source'/f'{name}.mpc').write_text(source)
    (target/'program.mpc').write_text(source); (target/'assembly').mkdir()
    job={'name':name,'mpspdz':str(root),'variant':variant,'task':workload,'public':cfg,
         'status':'compiling','timeout_seconds':cfg['timeout_seconds'],
         'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
         'environment':environment(root),'counts':operation_counts(cfg,variant)}
    write_json(target/'job.json',job)
    env=os.environ.copy(); env['SPARSE_PCA_PUBLIC_CONFIG']=str((inputs/'public.json').resolve())
    start=time.perf_counter()
    with (target/'compile.log').open('w') as log:
        process=subprocess.run([sys.executable,'compile.py','-R','128','-a',
            str((target/'assembly/code').resolve()),name],cwd=root,env=env,
            stdout=log,stderr=subprocess.STDOUT,timeout=600)
    job['compile_seconds']=time.perf_counter()-start
    job['status']='compiled' if process.returncode==0 else 'compile_failed'
    write_json(target/'job.json',job)
    if process.returncode: raise RuntimeError('compile failed: '+str(target/'compile.log'))
    for kind,pattern in [('Schedules',name+'.sch'),('Bytecode',name+'-*.bc')]:
        dest=target/'compiled'/kind; dest.mkdir(parents=True)
        for path in (root/'Programs'/kind).glob(pattern): shutil.copy2(path,dest/path.name)
    write_json(target/'compiler-evidence.json',compiler_evidence(target))
    return target.name


def prepare_experiment(cfg,directory,mpspdz,seeds=None,edges=True):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    audit_api(mpspdz,directory)
    plan={'config':cfg,'fixtures':[],'state':'compiling','seeds':seeds or cfg['seeds']}
    planfile=directory/'experiment.json'
    if planfile.exists():
        plan=json.loads(planfile.read_text())
        if plan['config']!=cfg or plan['seeds']!=(seeds or cfg['seeds']): raise ValueError('configuration changed')
    for name,parts,seed,main in fixtures(cfg,plan['seeds'],edges):
        print('Preparing',name,flush=True); fixture=directory/name
        if not (fixture/'fixture.json').exists(): prepare_fixture(parts,seed,cfg,fixture)
        entry=next((e for e in plan['fixtures'] if e['directory']==name),None)
        if entry is None:
            entry={'directory':name,'main':main,'jobs':[]}; plan['fixtures'].append(entry)
        # All correctness validation jobs are run before any measurement jobs.
        workloads=['validation','single','batch'] if main else ['validation']
        if main and abs(np.count_nonzero(np.vstack(parts))/np.vstack(parts).size-.1)<.002:
            workloads+=['pc1']
        for workload in workloads:
            for variant in VARIANTS:
                name_=variant+'-'+workload
                print('Compiling',name,name_,flush=True)
                entry['jobs'].append(name_) if name_ not in entry['jobs'] else None
                write_json(planfile,plan); compile_job(fixture,variant,workload,mpspdz)
        if main:
            legacy=fixture/'legacy_sparse-single'
            if not (legacy/'job.json').exists():
                # Compile the preserved program unchanged, with its original
                # secret coordinate input/reveal path. Diagnostic only.
                oldcfg={**cfg,'seed':seed,'capacity_per_hospital':json.loads((fixture/'fixture.json').read_text())['capacity']}
                write_json(fixture/'manifest.json',{'config':oldcfg})
                compile_program(fixture,legacy,mpspdz,'sparse_public','operator',True)
                oldjob=json.loads((legacy/'job.json').read_text())
                base=json.loads((fixture/'fixture.json').read_text())
                oldjob['public']={**base,'workload':'single','vector_count':1}
                oldjob['legacy_diagnostic']=True
                write_json(legacy/'job.json',oldjob)
            if legacy.name not in entry['jobs']: entry['jobs'].append(legacy.name)
            write_json(planfile,plan)
    plan['state']='compiled'; write_json(planfile,plan)
    evaluate_compiler(directory)
    return plan


def evaluate_compiler(directory):
    directory=Path(directory); plan=json.loads((directory/'experiment.json').read_text()); comparisons=[]
    for entry in plan['fixtures']:
        if not entry['main']: continue
        fixture=directory/entry['directory']
        for workload in ['validation','single','batch']:
            jobs={v:json.loads((fixture/(v+'-'+workload)/'job.json').read_text()) for v in VARIANTS}
            evidence={v:json.loads((fixture/(v+'-'+workload)/'compiler-evidence.json').read_text()) for v in VARIANTS}
            bits={v:evidence[v]['compiler_resources'].get('integer bits') for v in VARIANTS}
            scalar=jobs['sparse_scalar']; dot=jobs['sparse_dot']; n=dot['public']['vector_count']
            reductions=(scalar['counts']['matrix_scale_reductions_per_application']-
                        dot['counts']['matrix_scale_reductions_per_application'])*n
            # Installed TruncPrRing uses k+f+1 random bits per scale reduction.
            expected_saved_bits=reductions*(dot['public']['fixedpoint_bits']+dot['public']['fractional_bits']+1)
            actual_saved_bits=bits['sparse_grouped']-bits['sparse_dot']
            checks={'dot_reduction_savings_match_compiler':actual_saved_bits==expected_saved_bits,
                    'grouped_keeps_scalar_truncations':bits['sparse_scalar']==bits['sparse_grouped']}
            cfg=scalar['public']
            if cfg['realized_density']==1:
                checks['full_density_dot_matches_dense_resources']=(evidence['dense']['compiler_resources']==evidence['sparse_dot']['compiler_resources'])
            comparisons.append({'fixture':entry['directory'],'workload':workload,
                'saved_matrix_reductions':reductions,'expected_saved_random_bits':expected_saved_bits,
                'actual_saved_random_bits':actual_saved_bits,'checks':checks,
                'compiler_rounds':{v:evidence[v]['compiler_resources'].get('virtual machine rounds') for v in VARIANTS},
                'static_memory_instruction_counts':{v:{op:evidence[v]['assembly_instruction_counts'].get(op,0)
                    for op in ['ldms','stms','ldmsi','stmsi']} for v in VARIANTS},
                'note':'compiler rounds are not claimed as measured network rounds'})
    result={'passed':all(all(c['checks'].values()) for c in comparisons),'comparisons':comparisons}
    write_json(directory/'compiler-proof.json',result)
    if not result['passed']: raise RuntimeError('compiler proof differs from source expectations; inspect compiler-proof.json')
    return result


def parsed_output(text):
    rows={}; checksum=None
    for line in text.splitlines():
        if line.startswith('RESULT '):
            fields=line.split(); rows[int(fields[1])]=[float(x) for x in fields[2:]]
        if line.startswith('CHECKSUM '): checksum=[float(x) for x in line.split()[1:]]
    return rows,checksum


def error(actual,reference,atol,rtol):
    absolute=float(np.linalg.norm(np.asarray(actual)-reference)); norm=float(np.linalg.norm(reference))
    allowed=float(atol+rtol*norm)
    return {'absolute_error':absolute,'relative_error':absolute/max(norm,1e-12),
            'allowed_error':allowed,'passed':bool(absolute<=allowed)}


def policy_suffix(policy):
    if policy is None: return ''
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',policy['id']): raise ValueError('invalid policy identifier')
    if policy['absolute_tolerance']<0 or policy['relative_tolerance']<0: raise ValueError('negative tolerance')
    return '-'+policy['id']


def verify(directory,policy=None):
    directory=Path(directory); job=json.loads((directory/'job.json').read_text()); cfg=job['public']
    original_cfg=cfg
    if policy is not None:
        cfg={**cfg,'absolute_tolerance':policy['absolute_tolerance'],'relative_tolerance':policy['relative_tolerance']}
    result={'status':'awaiting_execution','accuracy_policy':policy or {'id':'initial','absolute_tolerance':cfg['absolute_tolerance'],'relative_tolerance':cfg['relative_tolerance']}}
    destination=directory/('verification'+policy_suffix(policy)+'.json')
    if not (directory/'execution.json').exists(): return result
    execution=json.loads((directory/'execution.json').read_text()); result['execution']=execution
    if execution['status']!='executed':
        result['status']='execution_failed'; write_json(destination,result); return result
    logs=[parse_log((directory/f'party-{i}.log').read_text()) for i in range(3)]
    for party,log in enumerate(logs):
        text=(directory/f'party-{party}.log').read_text()
        log['timer_communication_mb']={}
        for timer,mb in re.findall(r'^Time(\d+)\s*=\s*[\d.eE+-]+ seconds \(([\d.eE+-]+) MB',text,re.M):
            log['timer_communication_mb'][timer]=float(mb)
    with np.load(directory/'reference.npz') as data:
        xs=[np.vstack([data[f'party{i}'] for i in range(3)]),
            np.vstack([data[f'quantized{i}'] for i in range(3)])]
        vs=data['vectors'][:cfg['vector_count']]
        vectors_=[vs,quantize(vs,cfg['fractional_bits'])]
    actual,checksum=parsed_output((directory/'party-0.log').read_text())
    if job.get('legacy_diagnostic'):
        actual={0:logs[0].get('operator',[])}; checksum=logs[0].get('operator')
    checks={}; errors={}
    if cfg['workload']=='pc1':
        a=logs[0]; v=np.asarray(a.get('vector',[]))
        if v.shape!=(cfg['genes'],) or not np.isfinite(v).all(): checks['output']=False
        else:
            reference,lam=oracle(xs[0]); metrics=pc_metrics(xs[0],v,a['eigenvalue'],reference,lam)
            iv,_=power_iteration(xs[0],20,cfg['seed'])
            checks={'alignment':metrics['alignment']>.999,'eigenvalue':metrics['eigenvalue_relative_error']<.01,
                    'matched_iterations':abs(v@iv)/np.linalg.norm(v)>.999,'norm':abs(np.linalg.norm(v)-1)<.01}
            errors['pc1']=metrics
    else:
        for label,x,vectors__ in zip(['original','quantized'],xs,vectors_):
            z=x-x.mean(0); refs=np.array([z.T@(z@v) for v in vectors__])
            if cfg['workload'] in ('validation','single'):
                valid=set(actual)==set(range(len(refs))) and all(len(a)==cfg['genes'] and np.isfinite(a).all() for a in actual.values())
                checks[label+'_outputs']=valid
                if valid:
                    errs=[error(actual[i],ref,cfg['absolute_tolerance'],cfg['relative_tolerance']) for i,ref in enumerate(refs)]
                    for e,ref in zip(errs,refs):
                        initial_budget=original_cfg['absolute_tolerance']+original_cfg['relative_tolerance']*float(np.linalg.norm(ref))
                        e.update(initial_allowed_error=initial_budget,initial_passed=e['absolute_error']<=initial_budget)
                    errors[label]=errs; checks[label+'_accuracy']=all(e['passed'] for e in errs)
            # A checksum only verifies aggregate consumption, not each vector.
            # Per-vector accuracy is gated by the preceding validation workload.
            expected=sum((i+1)*ref for i,ref in enumerate(refs)); weights=sum(range(1,len(refs)+1))
            valid=checksum is not None and len(checksum)==cfg['genes'] and np.isfinite(checksum).all()
            checks[label+'_checksum_present']=valid
            if valid:
                budget=cfg['absolute_tolerance']*weights+cfg['relative_tolerance']*sum((i+1)*np.linalg.norm(ref) for i,ref in enumerate(refs))
                e=error(checksum,expected,budget,0); errors[label+'_checksum']=e
                checks[label+'_checksum']=e['passed']
    result.update(status='passed' if checks and all(checks.values()) else 'failed',
                  checks={k:bool(v) for k,v in checks.items()},errors=errors,parties=logs)
    write_json(destination,result); return result


def stage_and_run(compiled,trial,fixture,mpspdz,port,policy=None):
    trial=Path(trial); compiled=Path(compiled)
    if not (trial/'execution.json').exists():
        trial.mkdir(parents=True,exist_ok=True)
        for name in ['job.json','program.mpc']: shutil.copy2(compiled/name,trial/name)
        for name in ['inputs','compiled']: shutil.copytree(compiled/name,trial/name,dirs_exist_ok=True)
        shutil.copy2(Path(fixture)/'input.npz',trial/'reference.npz')
        print('Running',trial,flush=True); run_job(trial,mpspdz,port,precise_wall=True)
    return verify(trial,policy)


def run_experiment(directory,mpspdz,port=19000,policy=None):
    directory=Path(directory); plan=json.loads((directory/'experiment.json').read_text()); cfg=plan['config']
    if plan['state']!='compiled': raise ValueError('compilation is incomplete')
    evaluate_compiler(directory)
    # Gate the entire experiment on all density and edge fixtures, all vectors,
    # both references, and every implementation before collecting timings.
    gate_results=[]
    for entry in plan['fixtures']:
        fixture=directory/entry['directory']
        for variant in VARIANTS:
            result=stage_and_run(fixture/(variant+'-validation'),fixture/('check-'+variant),fixture,mpspdz,port,policy)
            gate_results.append({'fixture':entry['directory'],'variant':variant,'status':result['status'],
                                 'verification':str(fixture/('check-'+variant)/('verification'+policy_suffix(policy)+'.json'))})
            write_json(directory/('correctness-summary'+policy_suffix(policy)+'.json'),gate_results)
    if any(r['status']!='passed' for r in gate_results):
        raise RuntimeError('correctness gates failed; no performance data collected. Inspect '+str(directory/('correctness-summary'+policy_suffix(policy)+'.json')))
    for entry in plan['fixtures']:
        if not entry['main']: continue
        fixture=directory/entry['directory']; measurements=[]
        for workload in ['single','batch']:
            rng=random.Random(json.loads((fixture/'fixture.json').read_text())['seed']+991)
            for repetition in range(cfg['warmups']+cfg['repetitions']):
                order=list(VARIANTS)+(['legacy_sparse'] if workload=='single' else []); rng.shuffle(order)
                for variant in order:
                    trial=fixture/f'{workload}-{variant}-trial-{repetition}'
                    result=stage_and_run(fixture/(variant+'-'+workload),trial,fixture,mpspdz,port,policy)
                    execution=result['execution']; parties=result.get('parties',[])
                    row={'variant':variant,'workload':workload,'repetition':repetition,
                         'accuracy_policy':policy['id'] if policy else 'initial',
                         'warmup':repetition<cfg['warmups'],'status':result['status'],
                         'process_wall_seconds':execution['wall_seconds']}
                    row['wall_clock_method']=execution.get('wall_clock_method','legacy POSIX timeout polling; wall comparisons need rerun')
                    meta=json.loads((fixture/'fixture.json').read_text())
                    preparation=0. if variant=='legacy_sparse' else meta['schedule_seconds']
                    row['schedule_seconds']=preparation
                    row['complete_seconds']=execution['wall_seconds']+preparation
                    compiled_job=json.loads((fixture/(variant+'-'+workload)/'job.json').read_text())
                    row['cold_workflow_seconds']=row['complete_seconds']+compiled_job['compile_seconds']
                    if parties:
                        timers=parties[0]['timers']; row.update({f'timer_{k}':v for k,v in timers.items()})
                        row['kernel_seconds']=sum(timers.get(k,0) for k in ['3','4'])
                        row['operator_region_seconds']=timers.get('8',0)
                        if all('sent_mb' in p for p in parties): row['bytes_sent']=sum(p['sent_mb'] for p in parties)*1e6
                        if all('peak_rss_bytes' in p for p in parties): row['max_party_rss_bytes']=max(p['peak_rss_bytes'] for p in parties)
                        for i,p in enumerate(parties):
                            if 'peak_rss_bytes' in p: row[f'party_{i}_rss_bytes']=p['peak_rss_bytes']
                        if all(all(k in p['timer_communication_mb'] for k in ['3','4']) for p in parties):
                            row['kernel_bytes_sent']=sum(p['timer_communication_mb'][k] for p in parties for k in ['3','4'])*1e6
                        if all('8' in p['timer_communication_mb'] for p in parties):
                            row['operator_region_bytes_sent']=sum(p['timer_communication_mb']['8'] for p in parties)*1e6
                    measurements.append(row); write_json(fixture/'measurements.json',measurements)
                    if result['status']!='passed':
                        raise RuntimeError('measurement accuracy failed; investigate before continuing: '+str(trial))
        # PC1 is a separate confirmation, after operator collection.
    for entry in plan['fixtures']:
        fixture=directory/entry['directory']
        if 'dense-pc1' not in entry['jobs']: continue
        for variant in VARIANTS:
            result=stage_and_run(fixture/(variant+'-pc1'),fixture/('pc1-check-'+variant),fixture,mpspdz,port,policy)
            if result['status']!='passed': raise RuntimeError('PC1 confirmation failed: '+str(fixture))
    return summarize(directory)


def paired_speedup(base,candidate,samples=10000,seed=0):
    rounds=sorted(set(base)&set(candidate)); n=len(rounds)
    if n<2: return {'n_pairs':n,'status':'insufficient_pairs'}
    a=np.array([base[r] for r in rounds]); b=np.array([candidate[r] for r in rounds])
    if np.any(b<=0): return {'n_pairs':n,'status':'invalid_timer'}
    rng=np.random.default_rng(seed); indices=rng.integers(0,n,(samples,n))
    bootstrap=np.median(a[indices],axis=1)/np.median(b[indices],axis=1)
    ratio=float(np.median(a)/np.median(b)); lo,hi=np.quantile(bootstrap,[.025,.975])
    return {'n_pairs':n,'speedup':ratio,'ci95':[float(lo),float(hi)],
            'convincing_10_percent_reduction':bool(ratio>=1/.9 and lo>1),
            'method':'paired resampling of repetition rounds; ratio of medians'}


def fork_measurements(source,destination):
    """New timing population; reuse immutable compiled jobs and validation logs."""
    source=Path(source).resolve(); destination=Path(destination).resolve()
    if destination.exists(): raise ValueError('destination exists; do not overwrite timing evidence')
    plan=json.loads((source/'experiment.json').read_text())
    if plan['state']!='compiled': raise ValueError('source compilation is incomplete')
    destination.mkdir()
    plan['parent_experiment']=str(source)
    plan['measurement_clock']='concurrent blocking waitpid; no timeout polling'
    write_json(destination/'experiment.json',plan)
    for entry in plan['fixtures']:
        old=source/entry['directory']; new=destination/entry['directory']; new.mkdir()
        for name in ['input.npz','fixture.json','local-checks.json','manifest.json']:
            if (old/name).exists(): shutil.copy2(old/name,new/name)
        for name in entry['jobs']:
            # Only immutable compiled jobs are linked, never measurement outputs.
            (new/name).symlink_to(old/name,target_is_directory=True)
        for check in old.glob('check-*'):
            target=new/check.name; target.mkdir()
            for name in ['job.json','execution.json','reference.npz','party-0.log','party-1.log','party-2.log']:
                shutil.copy2(check/name,target/name)
            for name in check.glob('verification*.json'): shutil.copy2(name,target/name.name)
        # PC1 code is unchanged; preserve its already verified confirmation.
        for check in old.glob('pc1-check-*'):
            target=new/check.name; target.mkdir()
            for name in ['job.json','execution.json','reference.npz','party-0.log','party-1.log','party-2.log']:
                shutil.copy2(check/name,target/name)
            for name in check.glob('verification*.json'): shutil.copy2(name,target/name.name)
    if (source/'api-audit.json').exists(): shutil.copy2(source/'api-audit.json',destination/'api-audit.json')
    evaluate_compiler(destination)
    return {'prepared':str(destination),'parent_preserved':str(source),'measurements_copied':0}


def summarize(directory):
    directory=Path(directory); plan=json.loads((directory/'experiment.json').read_text()); cfg=plan['config']
    rows=[]; comparisons=[]
    for entry in plan['fixtures']:
        fixture=directory/entry['directory']; p=fixture/'measurements.json'
        if not p.exists(): continue
        measurements=json.loads(p.read_text()); meta=json.loads((fixture/'fixture.json').read_text())
        for workload in ['single','batch']:
            variants=list(VARIANTS)+(['legacy_sparse'] if workload=='single' else [])
            groups={variant:[m for m in measurements if m['variant']==variant and m['workload']==workload and not m['warmup']] for variant in variants}
            for variant,trials in groups.items():
                good=[t for t in trials if t['status']=='passed']
                row={'fixture':entry['directory'],'seed':meta['seed'],'density':meta['realized_density'],
                     'sparsity':1-meta['realized_density'],'nnz':meta['nnz'],'padded_slots':meta['padded_slots'],
                     'variant':variant,'workload':workload,'n_passed':len(good),'n_failed':len(trials)-len(good),
                     'schedule_seconds':0. if variant=='legacy_sparse' else meta['schedule_seconds']}
                policies={t.get('accuracy_policy','initial') for t in trials}
                if len(policies)>1: raise ValueError('mixed accuracy policies in one timing population')
                row['accuracy_policy']=next(iter(policies),'awaiting_execution')
                clocks={t.get('wall_clock_method','legacy POSIX timeout polling') for t in trials}
                if len(clocks)>1: raise ValueError('mixed process clocks in one timing population')
                row['wall_clock_method']=next(iter(clocks),'awaiting_execution')
                job=json.loads((fixture/(variant+'-'+workload)/'job.json').read_text())
                row['compile_seconds']=job['compile_seconds']
                for metric in ['complete_seconds','cold_workflow_seconds','process_wall_seconds','kernel_seconds','operator_region_seconds','bytes_sent','kernel_bytes_sent','operator_region_bytes_sent','max_party_rss_bytes']+['timer_'+str(k) for k in range(1,9)]+[f'party_{i}_rss_bytes' for i in range(3)]:
                    values=[t[metric] for t in good if metric in t]
                    if values:
                        q=np.quantile(values,[.25,.5,.75]); row[metric+'_median']=float(q[1]); row[metric+'_q1']=float(q[0]); row[metric+'_q3']=float(q[2])
                rows.append(row)
                for baseline in ['dense','sparse_scalar']:
                    if baseline==variant or variant=='dense': continue
                    for metric in ['complete_seconds','cold_workflow_seconds','kernel_seconds','bytes_sent','kernel_bytes_sent']:
                        a={t['repetition']:t[metric] for t in groups[baseline] if t['status']=='passed' and metric in t}
                        b={t['repetition']:t[metric] for t in good if metric in t}
                        item=paired_speedup(a,b,cfg['bootstrap_samples'],meta['seed'])
                        # Incomplete comparisons are recorded, never promoted to evidence.
                        if item['n_pairs']!=cfg['repetitions']: item['convincing_10_percent_reduction']=False
                        communication=metric in ['bytes_sent','kernel_bytes_sent']
                        outcome=('loading_path_diagnostic' if variant=='legacy_sparse' else
                                 'communication_benefit' if communication else
                                 'runtime_sparsity_benefit' if baseline=='dense' else 'implementation_improvement')
                        item['outcome']=outcome
                        item['supports_outcome']=(bool(item.get('speedup',0)>1 and item.get('ci95',[0])[0]>1
                              and item['n_pairs']==cfg['repetitions']) if communication else
                              item.get('convincing_10_percent_reduction',False))
                        if metric in ['complete_seconds','cold_workflow_seconds'] and 'blocking waitpid' not in row['wall_clock_method']:
                            item['supports_outcome']=False
                            item['convincing_10_percent_reduction']=False
                            item['timing_validity']='timeout-polling bias; do not interpret wall speedup'
                        comparisons.append({'fixture':entry['directory'],'workload':workload,'variant':variant,
                            'baseline':baseline,'metric':metric,**item})
    for name,data in [('summary',rows),('comparisons',comparisons)]:
        write_json(directory/(name+'.json'),data)
        if data:
            with (directory/(name+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(d.keys() for d in data))))
                writer.writeheader(); writer.writerows(data)
    return {'summary_rows':len(rows),'comparisons':len(comparisons),'measurements':'awaiting_execution' if not rows else 'collected; inspect correctness gates and pair counts'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['compile','run','summarize','bundle','fork'])
    p.add_argument('--config',default='configs/public-kernel.json'); p.add_argument('--out',default='results/public-kernel-optimized')
    p.add_argument('--mpspdz',default='/tmp/MP-SPDZ'); p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--port',type=int,default=19000)
    p.add_argument('--accuracy-policy',help='explicit alternate validation policy; preserves original verification files')
    p.add_argument('--from-out',default='results/public-kernel-optimized')
    args=p.parse_args(argv)
    try:
        if args.action=='compile': result=prepare_experiment(json.loads(Path(args.config).read_text()),args.out,args.mpspdz,args.seeds)
        elif args.action=='run':
            policy=json.loads(Path(args.accuracy_policy).read_text()) if args.accuracy_policy else None
            result=run_experiment(args.out,args.mpspdz,args.port,policy)
        elif args.action=='summarize': result=summarize(args.out)
        elif args.action=='fork': result=fork_measurements(args.from_out,args.out)
        else:
            from .cli import bundle
            result=str(bundle(args.out))
        print(json.dumps(result,indent=2)); return 0
    except (ValueError,RuntimeError,OSError,subprocess.SubprocessError) as exc:
        print(str(exc),file=sys.stderr); return 1


if __name__=='__main__': raise SystemExit(main())
