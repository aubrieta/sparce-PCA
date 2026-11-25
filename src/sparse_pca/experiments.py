"""Compile a controlled sweep separately from Terminal-only execution."""
import csv
import itertools
import json
from pathlib import Path
import random
import shutil
import statistics
from .data import prepare
from .runner import VARIANTS,compile_program,run_job,verify_job,write_json
from .plotting import report,benchmark_summary


def compile_sweep(config,directory,mpspdz,variants=VARIANTS,task='operator'):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    plan={'task':task,'cases':[],'warmups':config.get('warmups',1),
          'repetitions':config.get('repetitions',5)}
    sweep=config['sweep']
    for index,(shape,density,support,fractions) in enumerate(itertools.product(
            sweep['shapes'],sweep['densities'],sweep['supports'],sweep['partitions'])):
        cfg={k:v for k,v in config.items() if k!='sweep'}
        cfg.update(cells=shape[0],genes=shape[1],density=density,support=support,fractions=fractions)
        # A declared total-NNZ bound is valid for every hospital partition.
        # At high density cap it by the public maximum party matrix size.
        counts=[int(shape[0]*f) for f in fractions]
        counts[-1]+=shape[0]-sum(counts)
        cfg['capacity_per_hospital']=min(max(counts)*shape[1],max(1,round(shape[0]*shape[1]*density)))
        case=directory/f'case-{index:03d}'
        manifest=prepare(cfg,case/'data')
        entry={'directory':case.name,'config':cfg,'shape':manifest['shape'],
               'realized_density':manifest['density'],'sparsity':1-manifest['density'],
               'nnz':manifest['nnz'],'padded_entries':manifest['padded_entries'],
               'jobs':[],'status':'compiling'}
        plan['cases'].append(entry); write_json(directory/'experiment.json',plan)
        for variant in variants:
            target=case/f'{variant}-{task}'
            print(f"Compiling {case.name} {variant}: sparsity={entry['sparsity']:.4f}, capacity={cfg['capacity_per_hospital']}",flush=True)
            compile_program(case/'data',target,mpspdz,variant,task,task=='operator')
            entry['jobs'].append(target.name)
            write_json(directory/'experiment.json',plan)
        entry['status']='compiled'; write_json(directory/'experiment.json',plan)
    return plan


def run_sweep(directory,mpspdz,port=19000):
    directory=Path(directory); plan=json.loads((directory/'experiment.json').read_text())
    all_passed=True
    for case in plan['cases']:
        if case['status']!='compiled': raise ValueError('experiment contains incomplete compilation')
        root=directory/case['directory']; measurements=[]
        rng=random.Random(case['config']['seed'])
        # Interleave variants across repetition rounds to reduce thermal/order bias.
        for i in range(plan['warmups']+plan['repetitions']):
            jobs=list(case['jobs']); rng.shuffle(jobs)
            for name in jobs:
                compiled=root/name; job=json.loads((compiled/'job.json').read_text())
                trial=root/f"{job['variant']}-trial-{i}"
                if not (trial/'execution.json').exists():
                    trial.mkdir(parents=True,exist_ok=True)
                    for filename in ('job.json','input.npz','manifest.json'):
                        shutil.copy2(compiled/filename,trial/filename)
                    for folder in ('compiled','inputs'):
                        shutil.copytree(compiled/folder,trial/folder,dirs_exist_ok=True)
                    print(f"Running {case['directory']} {job['variant']} trial {i}",flush=True)
                    run_job(trial,mpspdz,port)
                verified=verify_job(trial)
                row={'variant':job['variant'],'repetition':i,'warmup':i<plan['warmups'],
                     'status':verified['status'],'wall_seconds':verified['execution']['wall_seconds']}
                if row['status']!='passed': all_passed=False
                parties=verified.get('parties',[])
                if parties and all('sent_mb' in p for p in parties):
                    row['total_sent_mb']=sum(p['sent_mb'] for p in parties)
                if parties and all(k in parties[0]['timers'] for k in ['3','4']):
                    row['kernel_seconds']=sum(parties[0]['timers'][k] for k in ['3','4'])
                measurements.append(row)
                write_json(root/'measurements.json',measurements)
                print(verified['status'],flush=True)
        benchmark_summary(root,measurements); report(root)
    summarize_sweep(directory)
    return all_passed


def summarize_sweep(directory):
    directory=Path(directory); plan=json.loads((directory/'experiment.json').read_text()); rows=[]
    for case in plan['cases']:
        path=directory/case['directory']/'measurements.json'
        measurements=json.loads(path.read_text()) if path.exists() else []
        variants=[json.loads((directory/case['directory']/name/'job.json').read_text())['variant'] for name in case['jobs']]
        per_case=[]
        for variant in variants:
            measured=[m for m in measurements if m['variant']==variant and not m['warmup']]
            good=[m for m in measured if m['status']=='passed']
            row={'case':case['directory'],'cells':case['shape'][0],'genes':case['shape'][1],
                 'sparsity':case['sparsity'],'density':case['realized_density'],
                 'variant':variant,'nnz':case['nnz'],'padded_entries':case['padded_entries'],
                 'n_passed':len(good),'n_failed':len(measured)-len(good),
                 'status':'passed' if len(good)==plan['repetitions'] else ('awaiting_execution' if not measured else 'incomplete_or_failed')}
            for field in ['wall_seconds','kernel_seconds','total_sent_mb']:
                values=[m[field] for m in good if field in m]
                if values:
                    row['median_'+field]=statistics.median(values)
                    row['min_'+field]=min(values); row['max_'+field]=max(values)
            per_case.append(row)
        dense=next((r for r in per_case if r['variant']=='dense' and r['status']=='passed'),None)
        for row in per_case:
            if dense and row['status']=='passed':
                for field in ['wall_seconds','kernel_seconds']:
                    if 'median_'+field in row and row['median_'+field]>0:
                        row[field+'_speedup']=dense['median_'+field]/row['median_'+field]
        rows.extend(per_case)
    write_json(directory/'sparsity-summary.json',rows)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (directory/'sparsity-summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows
