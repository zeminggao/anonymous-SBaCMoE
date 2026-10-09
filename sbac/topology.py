"""Explicit measured-bottleneck configuration; no inferred rank locality."""
import json,warnings
from pathlib import Path
import numpy as np

def load_topology(path):
    config=json.loads(Path(path).read_text())
    world=config['world_size'];sides=config['rank_to_bottleneck_side'];weights=config['inverse_bandwidth_weights']
    if not isinstance(world,int) or world<=0 or 64%world:raise ValueError('EP size must divide 64 experts')
    if len(sides)!=world or set(sides)!={0,1}:raise ValueError('Assign every global rank to side 0 or 1 of the measured bottleneck cut')
    if len(weights)!=2 or not np.isfinite(weights).all() or min(weights)<=0:raise ValueError('Provide two finite positive inverse-bandwidth weights')
    if config.get('example_only',False):warnings.warn('Example topology: calibrate cross-NUMA AND inter-node bandwidth before interpreting gains.',stacklevel=2)
    return config
