"""Plan one complete lookahead window from aligned arrays in an NPZ file."""
import argparse, json
import numpy as np
from sbac import regroup

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',help='NPZ containing a, b [sequences,layers] and tokens [sequences,length]')
    p.add_argument('--output',default='schedule.npz')
    p.add_argument('--epsilon',type=float,default=.02)
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--rank-nodes',type=int,nargs='+',default=[0,0,0,1])
    p.add_argument('--iterations',type=int)
    args=p.parse_args();data=np.load(args.input,allow_pickle=False)
    order,audit=regroup(data['a'],data['b'],data['tokens'],microbatch=args.microbatch,
                       rank_nodes=args.rank_nodes,epsilon=args.epsilon,iterations=args.iterations)
    np.savez(args.output,schedule=order)
    with open(args.output+'.json','w') as f:json.dump(audit,f,indent=2)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
