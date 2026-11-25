import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys
import tarfile
from .data import prepare
from .runner import VARIANTS, compile_program, compile_suite, run_job, verify_job, write_json
from .validation import validate_data
from .plotting import report, benchmark_summary, plot_sweep
from .experiments import compile_sweep,run_sweep,summarize_sweep


def run_suite(directory,mpspdz,port):
    directory=Path(directory)
    suite=json.loads((directory/'suite.json').read_text())
    # Resolve portable relative entries against the suite location.
    results=[]
    for item in suite['jobs']:
        job=Path(item); job=job if job.is_absolute() else directory/job
        print('Running',job,flush=True)
        run_job(job,mpspdz,port)
        result=verify_job(job); results.append(result)
        print(result['status'],flush=True)
    report(directory)
    return all(r['status']=='passed' for r in results)


def benchmark(config,destination,mpspdz,variants,task,port):
    destination=Path(destination); prepare(config,destination/'data')
    measurements=[]
    for variant in variants:
        compiled=destination/f'{variant}-{task}'
        compile_program(destination/'data',compiled,mpspdz,variant,task,task=='operator')
        for repetition in range(config.get('warmups',1)+config.get('repetitions',5)):
            trial=destination/f'{variant}-trial-{repetition}'
            trial.mkdir(parents=True,exist_ok=False)
            for name in ('job.json','input.npz','manifest.json'):
                shutil.copy2(compiled/name,trial/name)
            for name in ('compiled','inputs'): shutil.copytree(compiled/name,trial/name)
            execution=run_job(trial,mpspdz,port)
            verified=verify_job(trial)
            measurements.append({'variant':variant,'repetition':repetition,
                'warmup':repetition<config.get('warmups',1),
                'status':verified['status'],'wall_seconds':execution['wall_seconds']})
            write_json(destination/'measurements.json',measurements)
    report(destination)
    return benchmark_summary(destination,measurements)


def bundle(directory):
    directory=Path(directory).resolve(); target=directory/'return-logs.tar.gz'
    # Explicit allowlist: never include inputs, certificates, private keys, or runtime data.
    names={'job.json','compile.log','party-0.log','party-1.log','party-2.log',
           'execution.json','verification.json','summary.json','summary.csv',
           'local-validation.json','benchmark-summary.json','measurements.json','ssl.log',
           'experiment.json','sparsity-summary.json','sparsity-summary.csv',
           'api-audit.json','compiler-evidence.json','fixture.json','local-checks.json',
           'comparisons.json','comparisons.csv','compiler-proof.json','correctness-summary.json'}
    with tarfile.open(target,'w:gz') as archive:
        for p in sorted(directory.rglob('*')):
            policy_log=p.name.startswith(('verification-','correctness-summary-')) and p.suffix=='.json'
            if p.is_file() and (p.name in names or policy_log) and 'runtime' not in p.relative_to(directory).parts:
                archive.add(p,arcname=str(p.relative_to(directory)))
    return target


def main(argv=None):
    parser=argparse.ArgumentParser(description='Centered PCA reference and MP-SPDZ experiments')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','validate','compile-suite','benchmark','sweep'):
        p=sub.add_parser(name)
        p.add_argument('--config',default='configs/smoke.json')
        p.add_argument('--out',default='results/smoke')
        if name in ('compile-suite','benchmark','sweep'):
            p.add_argument('--mpspdz',default='/tmp/MP-SPDZ')
            p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
        if name in ('benchmark','sweep'):
            p.add_argument('--task',choices=['operator','pc1'],default='operator')
            p.add_argument('--port',type=int,default=19000)
    for name in ('run-suite','report','bundle','plot'):
        p=sub.add_parser(name); p.add_argument('--out',default='results/smoke')
        if name=='run-suite':
            p.add_argument('--mpspdz',default='/tmp/MP-SPDZ'); p.add_argument('--port',type=int,default=19000)
    p=sub.add_parser('compile-sweep')
    p.add_argument('--config',default='configs/sparsity-pilot.json')
    p.add_argument('--out',default='results/sparsity-pilot')
    p.add_argument('--mpspdz',default='/tmp/MP-SPDZ')
    p.add_argument('--task',choices=['operator','pc1'],default='operator')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p=sub.add_parser('run-sweep')
    p.add_argument('--out',default='results/sparsity-pilot')
    p.add_argument('--mpspdz',default='/tmp/MP-SPDZ')
    p.add_argument('--port',type=int,default=19000)
    p=sub.add_parser('summarize-sweep'); p.add_argument('--out',default='results/sparsity-pilot')
    args=parser.parse_args(argv)
    try:
        if args.command in ('prepare','validate','compile-suite','benchmark','sweep'):
            config=json.loads(Path(args.config).read_text())
            if args.command in ('prepare','validate','compile-suite'):
                prepare(config,Path(args.out)/'data')
        if args.command=='compile-sweep':
            result=compile_sweep(json.loads(Path(args.config).read_text()),args.out,args.mpspdz,args.variants,args.task)
        elif args.command=='run-sweep': return 0 if run_sweep(args.out,args.mpspdz,args.port) else 1
        elif args.command=='summarize-sweep': result=summarize_sweep(args.out)
        elif args.command=='prepare': result={'status':'prepared','data':str(Path(args.out)/'data')}
        elif args.command=='validate':
            result=validate_data(Path(args.out)/'data')
            write_json(Path(args.out)/'local-validation.json',result)
            print(json.dumps(result,indent=2)); return 0 if result['status']=='passed' else 1
        elif args.command=='compile-suite':
            result=compile_suite(Path(args.out)/'data',args.out,args.mpspdz,args.variants)
        elif args.command=='run-suite':
            return 0 if run_suite(args.out,args.mpspdz,args.port) else 1
        elif args.command=='report': result=report(args.out)
        elif args.command=='bundle': result=str(bundle(args.out))
        elif args.command=='plot': result=plot_sweep(args.out)
        elif args.command=='benchmark':
            result=benchmark(config,args.out,args.mpspdz,args.variants,args.task,args.port)
        elif args.command=='sweep':
            result=[]; sweep=config['sweep']
            for index,(shape,density,support,fractions) in enumerate(itertools.product(
                    sweep['shapes'],sweep['densities'],sweep['supports'],sweep['partitions'])):
                cfg={k:v for k,v in config.items() if k!='sweep'}
                cfg.update(cells=shape[0],genes=shape[1],density=density,support=support,fractions=fractions)
                # Declared synthetic total-NNZ bound, independent of realized allocation.
                cfg['capacity_per_hospital']=max(1,round(shape[0]*shape[1]*density))
                destination=Path(args.out)/f'case-{index:04d}'
                try: benchmark(cfg,destination,args.mpspdz,args.variants,args.task,args.port)
                except (ValueError,RuntimeError) as exc:
                    destination.mkdir(parents=True,exist_ok=True)
                    write_json(destination/'failure.json',{'status':'failed','error':str(exc)})
                result.append(str(destination))
        print(json.dumps(result,indent=2)); return 0
    except (OSError,ValueError,RuntimeError) as exc:
        print(str(exc),file=sys.stderr); return 1


if __name__=='__main__': raise SystemExit(main())
