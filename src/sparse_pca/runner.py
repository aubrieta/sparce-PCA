"""Compile programs here, execute party processes separately in a Terminal."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
from .data import load
from .export import export
from .reference import dense_operator, initial_vector, oracle, power_iteration
from .metrics import pc_metrics, relative_error

ROOT=Path(__file__).resolve().parents[2]
PINNED_REVISION="f5f0cd2433c61d464666cbae19da16ac197d7a75"
VARIANTS=("dense","sparse_public","sparse_private")


def prepare_certificates(runtime, log_path):
    """Generate and verify SAN-bearing identities; compatible with OpenSSL 4.

    MP-SPDZ's stock setup script only sets CN, which current OpenSSL no longer
    accepts for hostname verification. Keep verification enabled and fix names.
    """
    candidates=[Path('/opt/homebrew/opt/openssl@4/bin/openssl'),
                Path('/opt/homebrew/opt/openssl@3/bin/openssl')]
    found=shutil.which('openssl')
    if found: candidates.append(Path(found))
    executable=next((str(p) for p in candidates if p.is_file()),None)
    if executable is None: raise RuntimeError('OpenSSL is required for party certificates')
    data=Path(runtime).resolve()/'Player-Data'; data.mkdir(parents=True,exist_ok=True)
    def verify(party,log):
        certificate=data/f'P{party}.pem'
        if not certificate.exists() or not (data/f'P{party}.key').exists(): return False
        # Require SAN even when an older OpenSSL would still accept only CN.
        extensions=subprocess.run([executable,'x509','-in',str(certificate),'-noout',
                                   '-ext','subjectAltName'],capture_output=True,text=True)
        if f'DNS:P{party}' not in extensions.stdout: return False
        return subprocess.run([executable,'verify','-CApath',str(data),'-verify_hostname',
                               f'P{party}',str(certificate)],stdout=log,
                              stderr=subprocess.STDOUT).returncode==0
    with Path(log_path).open('w') as log:
        subprocess.run([executable,'version'],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not all(verify(party,log) for party in range(3)):
            for party in range(3):
                cmd=[executable,'req','-new','-x509','-sha256','-days','30',
                     '-subj',f'/CN=P{party}','-addext',f'subjectAltName=DNS:P{party}',
                     '-addext','basicConstraints=critical,CA:TRUE',
                     '-addext','keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign',
                     '-addext','extendedKeyUsage=serverAuth,clientAuth',
                     '-out',str(data/f'P{party}.pem')]
                key=data/f'P{party}.key'
                if key.exists(): cmd+=['-key',str(key)]
                else: cmd+=['-newkey','rsa:2048','-nodes','-keyout',str(key)]
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                key.chmod(0o600)
            for link in data.iterdir():
                if link.is_symlink() and re.fullmatch(r'[0-9a-f]{8}\.\d+',link.name): link.unlink()
            subprocess.run([executable,'rehash',str(data)],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not all(verify(party,log) for party in range(3)):
            raise RuntimeError(f'party certificate verification failed; see {log_path}')


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def environment(mpspdz):
    revision=subprocess.check_output(["git","-C",str(mpspdz),"rev-parse","HEAD"],text=True).strip()
    packages={}
    for name in ("numpy","scipy","scanpy","matplotlib"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return {"python":sys.version,"platform":platform.platform(),"packages":packages,
            "mpspdz_revision":revision,"pinned_revision":PINNED_REVISION,
            "revision_matches":revision==PINNED_REVISION,
            "protocol":"three-party semi-honest replicated ring; at most one corrupted party",
            "network":"localhost; three processes on one host"}


def compile_program(data_dir, destination, mpspdz, variant, task, validation=False):
    if variant not in VARIANTS or task not in ("operator","pc1"):
        raise ValueError("unknown variant or task")
    if task=="operator" and not validation:
        raise ValueError("operator task requires explicit validation mode")
    mpspdz=Path(mpspdz).resolve(); destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    parts,manifest=load(data_dir)
    cfg=manifest["config"]
    public,numeric=export(parts,cfg,destination/"inputs",variant,task,validation)
    if public["fractional_bits"]<1 or public["fixedpoint_bits"]<=public["fractional_bits"]:
        raise ValueError("fixed-point width must exceed a positive fractional width")
    if public["iterations"]<1:
        raise ValueError("iteration count must be positive")
    if public["fixedpoint_bits"]*2>public["ring_bits"]:
        raise ValueError("ring width must be at least twice fixed-point bit length")
    if public["zero_threshold"]**2 < 2**(-public["fractional_bits"]):
        raise ValueError("squared zero threshold must be representable at configured precision")
    name=f"spca_{variant}_{task}_{uuid.uuid4().hex[:12]}"
    source=(ROOT/"mpc/common.mpc").read_text()
    source_path=mpspdz/"Programs/Source"/(name+".mpc")
    # Retain an exact compiled-source snapshot beside the logs.
    (destination/"program.mpc").write_text(source)
    source_path.write_text(source)
    shutil.copy2(ROOT/"src/sparse_pca/oblivious.py",destination/"inputs/oblivious_network.py")
    shutil.copy2(Path(data_dir)/"input.npz",destination/"input.npz")
    write_json(destination/"manifest.json",manifest)
    metadata={"name":name,"mpspdz":str(mpspdz),"variant":variant,"task":task,
              "public":public,"numeric_bounds":numeric,"environment":environment(mpspdz),
              "timeout_seconds":cfg.get("timeout_seconds",300),
              "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
              "network_sha256":hashlib.sha256((ROOT/'src/sparse_pca/oblivious.py').read_bytes()).hexdigest(),
              "status":"compiling"}
    write_json(destination/"job.json",metadata)
    env=os.environ.copy(); env["SPARSE_PCA_PUBLIC_CONFIG"]=str(destination/"inputs/public.json")
    cmd=[sys.executable,"compile.py","-R",str(public["ring_bits"]),name]
    started=time.perf_counter()
    try:
        with (destination/"compile.log").open("w") as log:
            subprocess.run(cmd,cwd=mpspdz,env=env,stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=cfg.get("compile_timeout_seconds",600))
    except (subprocess.SubprocessError,OSError) as exc:
        metadata.update(status="compile_failed",error=str(exc))
        write_json(destination/"job.json",metadata)
        raise RuntimeError(f"compile failed; see {destination/'compile.log'}") from exc
    metadata.update(status="compiled",compile_seconds=time.perf_counter()-started)
    (destination/"compiled/Bytecode").mkdir(parents=True,exist_ok=True)
    (destination/"compiled/Schedules").mkdir(parents=True,exist_ok=True)
    schedule=mpspdz/"Programs/Schedules"/(name+".sch")
    shutil.copy2(schedule,destination/"compiled/Schedules"/schedule.name)
    for bytecode in (mpspdz/"Programs/Bytecode").glob(name+"-*.bc"):
        shutil.copy2(bytecode,destination/"compiled/Bytecode"/bytecode.name)
    write_json(destination/"job.json",metadata)
    return metadata


def run_job(directory, mpspdz=None, port=19000):
    """Runs only when explicitly invoked, never as part of compile or validate."""
    directory=Path(directory).resolve()
    job=json.loads((directory/"job.json").read_text())
    if job["status"]!="compiled": raise ValueError("job did not compile")
    mpspdz=Path(mpspdz or job["mpspdz"]).resolve()
    runtime=directory/"runtime"
    (runtime/"Programs/Bytecode").mkdir(parents=True,exist_ok=True)
    (runtime/"Programs/Schedules").mkdir(parents=True,exist_ok=True)
    (runtime/"Player-Data").mkdir(parents=True,exist_ok=True)
    # macOS MP-SPDZ links libSPDZ.so by a relative install name. Its stock
    # scripts run in the checkout; our isolated working directory needs the
    # same library available locally. A symlink also works through /usr/bin/time,
    # whose system loader can strip DYLD_* environment variables on macOS.
    for library in list(mpspdz.glob('*.so'))+list(mpspdz.glob('*.dylib')):
        target=runtime/library.name
        if target.is_symlink():
            if target.resolve()==library.resolve(): continue
            target.unlink()
        if not target.exists(): target.symlink_to(library.resolve())
    for kind in ("Bytecode","Schedules"):
        for path in (directory/"compiled"/kind).iterdir():
            shutil.copy2(path,runtime/"Programs"/kind/path.name)
    for path in (directory/"inputs").glob("Input-P*-0"):
        shutil.copy2(path,runtime/"Player-Data"/path.name)
    # Verify identities before starting parties; replace legacy CN-only certs.
    prepare_certificates(runtime,directory/'ssl.log')
    processes=[]; handles=[]; started=time.perf_counter()
    result={"status":"failed","port":port,"party_exit_codes":[],"peak_memory_note":
            "OS time utility measures each party process; aggregate memory is not measured",
            "environment":environment(mpspdz)}
    try:
        for party in range(3):
            handle=(directory/f"party-{party}.log").open("w"); handles.append(handle)
            cmd=[str(mpspdz/"replicated-ring-party.x"),str(party),job["name"],
                 "-pn",str(port),"-h","localhost","-v"]
            if Path("/usr/bin/time").exists():
                cmd=["/usr/bin/time","-l" if sys.platform=="darwin" else "-v"]+cmd
            processes.append(subprocess.Popen(cmd,cwd=runtime,stdout=handle,stderr=subprocess.STDOUT))
        deadline=started+job["timeout_seconds"]
        for p in processes:
            p.wait(timeout=max(0.01,deadline-time.perf_counter()))
        result["party_exit_codes"]=[p.returncode for p in processes]
        result["status"]="executed" if all(p.returncode==0 for p in processes) else "failed"
    except subprocess.TimeoutExpired:
        result["status"]="timeout"
    except OSError as exc:
        result["error"]=str(exc)
    finally:
        for p in processes:
            if p.poll() is None: p.kill()
        for p in processes: p.wait()
        for h in handles: h.close()
        result["wall_seconds"]=time.perf_counter()-started
        result["party_exit_codes"]=[p.returncode for p in processes]
        write_json(directory/"execution.json",result)
    return result


def parse_log(text):
    result={"timers":{}}
    for line in text.splitlines():
        if line.startswith("VALIDATION_OPERATOR "):
            result["operator"]=[float(x) for x in line.split()[1:]]
        elif line.startswith("PC1 "):
            result["vector"]=[float(x) for x in line.split()[1:]]
        elif line.startswith("EIGENVALUE "):
            result["eigenvalue"]=float(line.split()[1])
        match=re.search(r"Time(\d+)\s*=\s*([\d.eE+-]+)",line)
        if match: result["timers"][match[1]]=float(match[2])
        match=re.search(r"^Data sent = ([\d.eE+-]+) MB",line)
        if match: result["sent_mb"]=float(match[1])
        match=re.search(r"^Global data sent = ([\d.eE+-]+) MB",line)
        if match: result["global_sent_mb"]=float(match[1])
        match=re.search(r"(\d+)\s+maximum resident set size",line)
        if match: result["peak_rss_bytes"]=int(match[1])
        match=re.search(r"Maximum resident set size \(kbytes\): (\d+)",line)
        if match: result["peak_rss_bytes"]=1024*int(match[1])
    return result


def verify_job(directory):
    directory=Path(directory)
    job=json.loads((directory/"job.json").read_text())
    execution_path=directory/"execution.json"
    result={"status":"awaiting_execution","checks":{},"variant":job["variant"],"task":job["task"]}
    if not execution_path.exists():
        write_json(directory/"verification.json",result); return result
    execution=json.loads(execution_path.read_text())
    logs=[parse_log((directory/f"party-{i}.log").read_text(errors="replace")) for i in range(3)]
    result.update(execution=execution,parties=logs)
    if execution["status"]!="executed":
        result["status"]="execution_failed"
    else:
        parts,_=load(directory); x=np.vstack(parts); actual=logs[0]
        if job["task"]=="operator" and "operator" in actual:
            expected=dense_operator(x,initial_vector(x.shape[1],job["public"]["seed"]))
            vector=np.asarray(actual['operator'])
            valid=vector.shape==expected.shape and np.all(np.isfinite(vector))
            if valid:
                error=relative_error(vector,expected)
                result["metrics"]={"operator_relative_error":error}
                result["checks"]={"operator_error":error<1e-3}
            else:
                result['checks']={'valid_operator_output':False}
        elif "vector" in actual and "eigenvalue" in actual:
            v=np.asarray(actual["vector"])
            if v.shape!=(x.shape[1],) or not np.all(np.isfinite(v)) or np.linalg.norm(v)<1e-12:
                result["checks"]={"valid_output_vector":False}
            else:
                rv,rl=oracle(x)
                metrics=pc_metrics(x,v,actual["eigenvalue"],rv,rl)
                iv,il=power_iteration(x,job["public"]["iterations"],job["public"]["seed"])
                metrics["matched_iteration_alignment"]=float(abs(v@iv)/np.linalg.norm(v))
                result["metrics"]=metrics
                result["checks"]={"matched_iteration_alignment":metrics["matched_iteration_alignment"]>.999,
                    "eigenvalue_error":metrics["eigenvalue_relative_error"]<.01,
                    "svd_alignment":metrics["alignment"]>.999 or metrics["degenerate"],
                    "unit_norm":abs(np.linalg.norm(v)-1)<.01}
        else: result["checks"]={"output_present":False}
        result['checks']={key:bool(value) for key,value in result['checks'].items()}
        result["status"]="passed" if result["checks"] and all(result["checks"].values()) else "failed"
    write_json(directory/"verification.json",result)
    return result


def compile_suite(data_dir,destination,mpspdz,variants=VARIANTS):
    destination=Path(destination)
    jobs=[]
    for task in ("operator","pc1"):
        for variant in variants:
            target=destination/f"{variant}-{task}"
            compile_program(data_dir,target,mpspdz,variant,task,validation=task=="operator")
            jobs.append(target.name)
    write_json(destination/"suite.json",{"jobs":jobs})
    return jobs
