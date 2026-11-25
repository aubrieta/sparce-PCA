import csv
import json
from pathlib import Path
import statistics
from .runner import verify_job, write_json


def report(directory):
    directory=Path(directory)
    rows=[]
    for path in sorted(directory.rglob('job.json')):
        job=json.loads(path.read_text())
        if job['status']!='compiled':
            rows.append({'job':str(path.parent),'variant':job['variant'],'task':job['task'],
                         'status':job['status']}); continue
        result=verify_job(path.parent)
        row={'job':str(path.parent),'variant':job['variant'],'task':job['task'],
             'status':result['status'],'compile_seconds':job.get('compile_seconds')}
        row.update(result.get('metrics',{}))
        if 'execution' in result: row['wall_seconds']=result['execution']['wall_seconds']
        parties=result.get('parties',[])
        if parties:
            if all('sent_mb' in p for p in parties):
                row['total_sent_mb']=sum(p['sent_mb'] for p in parties)
            if all('peak_rss_bytes' in p for p in parties):
                row['max_party_rss_bytes']=max(p['peak_rss_bytes'] for p in parties)
            row.update({f'timer_{k}_seconds':v for k,v in parties[0]['timers'].items()})
        rows.append(row)
    columns=sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with (directory/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    write_json(directory/'summary.json',rows)
    return rows


def benchmark_summary(directory, measurements):
    groups={}
    for row in measurements:
        if row['warmup'] or row['status']!='passed': continue
        groups.setdefault(row['variant'],[]).append(row['wall_seconds'])
    stats={k:{'n':len(v),'median_seconds':statistics.median(v),
              'min_seconds':min(v),'max_seconds':max(v),
              'stdev_seconds':statistics.stdev(v) if len(v)>1 else 0.}
           for k,v in groups.items()}
    for variant,item in stats.items():
        if 'dense' in stats: item['speedup']=stats['dense']['median_seconds']/item['median_seconds']
    write_json(Path(directory)/'benchmark-summary.json',stats)
    return stats


def plot_sweep(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    points={}
    for p in directory.rglob('benchmark-summary.json'):
        data=p.parent/'data/manifest.json'
        if not data.exists(): continue
        manifest=json.loads(data.read_text()); cfg=manifest['config']
        for variant,stats in json.loads(p.read_text()).items():
            if 'speedup' in stats:
                key=(tuple(manifest['shape']),cfg.get('support','pbmc'),tuple(cfg.get('fractions',[])),variant)
                points.setdefault(key,[]).append((manifest['density'],stats['speedup']))
    figures=[]
    for index,(key,values) in enumerate(points.items()):
        values.sort(); fig,ax=plt.subplots()
        ax.plot([v[0] for v in values],[v[1] for v in values],marker='o')
        ax.axhline(1,color='gray',linestyle='--'); ax.set_xscale('log')
        ax.set(xlabel='Realized nonzero density',ylabel='Dense / sparse wall time',title=str(key))
        target=directory/f'speedup-{index}.png'; fig.savefig(target,bbox_inches='tight'); plt.close(fig)
        figures.append(str(target))
    return figures
