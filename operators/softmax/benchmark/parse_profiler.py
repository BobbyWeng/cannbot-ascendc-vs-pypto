#!/usr/bin/env python3
import os,sys,json,glob; from collections import defaultdict
def find_prof_dir(d):
    for sd in sorted(os.listdir(d),reverse=True):
        if sd.startswith('PROF_'): return os.path.join(d,sd)
    return None
def find_trace(d):
    for r,_,fs in os.walk(d):
        for f in fs:
            if f.startswith('msprof_') and f.endswith('.json'): return os.path.join(r,f)
    return None
def main():
    raw=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else None
    pd=find_prof_dir(raw)
    if not pd: print(f"error: no PROF_* in {raw}"); sys.exit(1)
    tp=find_trace(pd)
    if not tp: print(f"error: no msprof_*.json under {pd}"); sys.exit(1)
    with open(tp) as f: data=json.load(f)
    tr=data if isinstance(data,list) else data.get('traceEvents',[])
    ke=[]; he=[]
    for ev in tr:
        if not isinstance(ev,dict) or ev.get('ph')!='X': continue
        n=ev.get('name',''); d=ev.get('dur',0); tt=ev.get('args',{}).get('Task Type','N/A')
        if tt.startswith('KERNEL_'): ke.append({'name':n,'type':tt,'dur_us':d})
        elif tt=='N/A' and d>0: he.append({'name':n,'dur_us':d})
    td=sum(e['dur_us'] for e in ke)
    pm=max(ke,key=lambda e:e['dur_us']) if ke else {}
    bt=defaultdict(list)
    for e in ke: bt[e['type']].append(e['dur_us'])
    r={"kernel_count":len(ke),"kernels_per_call":round(len(ke)/100,2),
       "all_device_kernels_us":round(td/100,3),
       "primary_compute_kernel_us":round(pm.get('dur_us',0)/1,3),
       "primary_compute_type":pm.get('type',''),
       "kernel_names":sorted(set(e['name'] for e in ke)),
       "by_type":{k:{'count':len(v),'total_us':round(sum(v),2),'mean_us':round(sum(v)/len(v),3)} for k,v in bt.items()},
       "host_events_count":len(he)}
    if out:
        os.makedirs(os.path.dirname(out),exist_ok=True)
        with open(out,'w') as f: json.dump(r,f,indent=2)
        print(f"parsed -> {out}")
    else: print(json.dumps(r,indent=2))
if __name__=='__main__': main()
