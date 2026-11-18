"""Offline sequence evaluation. Only this stage reads the supplied reference route."""
import csv,json,itertools,sys
from run import ROOT,DATA
def edit(a,b):
    last=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):
            row.append(min(row[-1]+1,last[j]+1,last[j-1]+(x!=y)))
        last=row
    return last[-1]
def main(phase):
    folder=ROOT/'outputs'/phase;table=[]
    for path in sorted(folder.glob('*.json')):
        r=json.loads(path.read_text());ref=list(map(int,(DATA/(path.stem+'.route')).read_text().split()))
        for method in ['baseline','arc_id']:
            ids=[p[method] for p in r['points']]
            seq=[k for k,_ in itertools.groupby(ids) if k is not None]
            d=edit(seq,ref)
            table.append([path.stem,method,len(seq),len(ref),d,d/max(len(ref),1)])
    with (folder/'sequence_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['trip','method','observed_arcs','reference_entries','edit_distance','edits_per_reference_entry']);w.writerows(table)
if __name__=='__main__':main(sys.argv[1])
