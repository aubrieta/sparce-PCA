from pathlib import Path
import json
import numpy as np
from .reference import initial_vector
from .fixedpoint import bounds


def export(parts, config, destination, variant, task="pc1", validation=False):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    c=sum(map(len,parts)); g=parts[0].shape[1]; cap=config["capacity_per_hospital"]
    if any(np.count_nonzero(p)>cap for p in parts):
        raise ValueError("public sparse capacity exceeded")
    scale=config.get('input_scale',1.)
    if scale<=0 or not float(np.log2(scale)).is_integer():
        raise ValueError('input_scale must be a positive power of two')
    x=np.vstack(parts)
    numeric=bounds(x*scale,config.get("fractional_bits",20),config.get("ring_bits",128),
                   config.get("fixedpoint_bits",40))
    output_bound=max(1.,len(x)*x.shape[1]*float(np.max(np.abs(x),initial=0))**2)
    numeric['required_output_fixedpoint_bits']=int(np.ceil(np.log2(output_bound)))+config.get('fractional_bits',20)+3
    numeric['safe']=numeric['safe'] and numeric['required_output_fixedpoint_bits']<=config.get('fixedpoint_bits',40)
    if not numeric["safe"]:
        raise ValueError(f"unsafe arithmetic width: {numeric}")
    # All variants include local column sums as secret owner inputs.
    for party,x in enumerate(parts):
        x=x*scale
        values=x.sum(axis=0).tolist()
        if variant=="dense":
            values+=x.ravel().tolist()
        else:
            rows,cols=np.nonzero(x)
            triples=[[int(r),int(j),float(x[r,j])] for r,j in zip(rows,cols)]
            triples += [[0,0,0.]]*(cap-len(triples))
            values += [z for triple in triples for z in triple]
        (destination/f"Input-P{party}-0").write_text("\n".join(format(z,'.17g') for z in values)+"\n")
    public={"rows":[len(p) for p in parts],"genes":g,"cells":c,"capacity":cap,
            "variant":variant,"task":task,"validation":validation,
            "iterations":config.get("iterations",20),"seed":config["seed"],
            "fractional_bits":config.get("fractional_bits",20),
            "fixedpoint_bits":config.get("fixedpoint_bits",40),
            "ring_bits":config.get("ring_bits",128),
            "zero_threshold":config.get("zero_threshold",1e-8),
            "input_scale":scale,
            "initial_vector":initial_vector(g,config["seed"]).tolist(),
            "leakage":"coordinates and capacity" if variant=="sparse_public" else "dimensions and fixed capacities"}
    (destination/"public.json").write_text(json.dumps(public,indent=2))
    return public,numeric
