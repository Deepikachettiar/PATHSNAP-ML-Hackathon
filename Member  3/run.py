"""Member 3 directed road decoder. All coordinates in output are longitude, latitude."""
import argparse, csv, gzip, heapq, json, math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

def metres(a,b):
    x=(a[0]-b[0])*math.cos(math.radians((a[1]+b[1])/2))
    return 111195*math.hypot(x,a[1]-b[1])

def rows(trip,ext,cast):
    return [tuple(map(cast,line.split())) for line in (DATA/(trip+'.'+ext)).read_text().splitlines()]

def decode(trip, groups, track, nodes, arcs):
    graph=defaultdict(list)
    lengths=[]
    for arc,(u,v) in enumerate(arcs):
        length=metres(nodes[u],nodes[v])
        lengths.append(length)
        graph[u].append((v,length,arc))
    @lru_cache(maxsize=4096)
    def paths(start,bound):
        best={start:0}; prev={}; queue=[(0,start)]
        while queue:
            dist,u=heapq.heappop(queue)
            if dist!=best[u]:continue
            for v,w,arc in graph[u]:
                cost=dist+w
                if cost<=bound and cost<best.get(v,math.inf):
                    best[v]=cost;prev[v]=(u,arc);heapq.heappush(queue,(cost,v))
        return best,prev
    trace=[]; ends=[]; reasons=[]; scores={}; offsets={}
    for i,group in enumerate(groups):
        for arc,(pos,_) in group.items():
            offsets[i,arc]=min(lengths[arc],metres(nodes[arcs[arc][0]],pos))
        parents={}; new={}; reason=''
        dt=track[i][2]-track[i-1][2] if i else 0
        if group and scores and 0<dt<=30:
            step=metres(track[i],track[i-1]);bound=math.ceil(60*dt+40)
            for old,prior in scores.items():
                costs,back=paths(arcs[old][1],bound)
                options=[]
                for arc in group:
                    start=arcs[arc][0]
                    delta=offsets[i,arc]-offsets[i-1,old]
                    direct=old==arc and delta>=0
                    distance=delta if direct else lengths[old]-offsets[i-1,old]+costs.get(start,math.inf)+offsets[i,arc]
                    if distance<=bound:options.append((arc,-abs(distance-step)/25,direct))
                if not options:continue
                peak=max(w for _,w,_ in options)
                norm=peak+math.log(sum(math.exp(w-peak) for _,w,_ in options))
                for arc,w,direct in options:
                    value=prior+w-norm+group[arc][1]
                    if value>new.get(arc,-math.inf):
                        connectors=[];u=arcs[arc][0]
                        if not direct:
                            while u!=arcs[old][1]:
                                u,edge=back[u];connectors.append(edge)
                        parents[arc]=(old,list(reversed(connectors)),direct);new[arc]=value
        if not new:
            reason='missing_candidates_or_scores' if not group else ('start' if not i else 'chain_restart')
            new={arc:weight for arc,(_,weight) in group.items()}
        trace.append(parents);scores=new
        ends.append(max(scores,key=scores.get) if scores else None);reasons.append(reason)
    chosen=ends.copy()
    for i in range(len(trace)-1,0,-1):
        if chosen[i] in trace[i]:chosen[i-1]=trace[i][chosen[i]][0]
    points=[];lines=[];sequence=[]
    for i,arc in enumerate(chosen):
        group=groups[i];base=max(group,key=lambda a:group[a][1]) if group else None
        pos=group[arc][0] if group else None
        points.append(dict(point_index=i,arc_id=arc,baseline=base,lonlat=pos,status=reasons[i] or 'connected'))
        if arc is None:continue
        if arc in trace[i]:
            old,edges,direct=trace[i][arc]
            if not direct:
                lines[-1].extend([nodes[arcs[old][1]]]+[nodes[arcs[e][1]] for e in edges])
                sequence.extend(edges)
        else:lines.append([])
        lines[-1].append(pos)
        if not sequence or sequence[-1]!=arc or reasons[i]:sequence.append(arc)
    replay=[]
    fallback=[]
    for anchor,group in enumerate(groups):
        ranked=sorted(group,key=lambda a:-group[a][1])
        if len(ranked)<2 or group[ranked[0]][1]-group[ranked[1]][1]>0.5:continue
        for end in range(anchor,min(anchor+21,len(groups))):
            if end>anchor and reasons[end]:break
            arc=ends[end]
            for j in range(end,anchor,-1):arc=trace[j][arc][0]
            replay.append(dict(anchor=anchor,prefix_end=end,anchor_arc=arc,lonlat=group[arc][0],gps=track[end][:2]))
        if len({r["anchor_arc"] for r in replay})>1:break
        if len(replay)>len(fallback):fallback=replay.copy()
        replay.clear()
    replay=replay or fallback
    roads={}
    if replay:
        for arc in groups[replay[0]["anchor"]]:
            u,v=arcs[arc]; roads[arc]=[nodes[u],nodes[v]]
    return dict(anchor_roads=roads,trip_id=trip,points=points,lines=lines,replay=replay,sequence=sequence)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scores',type=Path,default=DATA/'scores.gz')
    parser.add_argument('--labels',type=Path,default=DATA/'derived_labels.csv',help='CSV: trip_id,point_index,reference_arc_id (eligible rows only)')
    args=parser.parse_args()
    scores={}
    if args.scores:
        opener=gzip.open if args.scores.suffix=='.gz' else open
        with opener(args.scores,'rt') as f:
            for r in csv.DictReader(f):
                k=(r['trip_id'],int(r['point_index']),int(r['arc_id']))
                prob=float(r['candidate_probability'])
                if k in scores or not 0<=prob<=1:raise ValueError(k)
                scores[k]=math.log(max(1e-300,prob))
    trips=defaultdict(lambda:defaultdict(dict))
    with gzip.open(DATA/'candidates.gz','rt') as f:
        for r in csv.DictReader(f):
            t=r['trip_id'];i=int(r['point_index']);a=int(r['arc_id'])
            weight=scores[t,i,a]
            group=trips[t][i]
            group[a]=((float(r['projected_lon']),float(r['projected_lat'])),weight)
    out=ROOT/'outputs'/'final';out.mkdir(parents=True,exist_ok=True)
    labels={}
    if args.labels:
        with args.labels.open() as f:
            for r in csv.DictReader(f):labels[r['trip_id'],int(r['point_index'])]=int(r['reference_arc_id'])
    summary=[]
    for t,table in sorted(trips.items()):
        track=rows(t,'track',float)
        groups=[table.get(i,{}) for i in range(len(track))]
        groups=[dict(sorted(g.items())) if all(w is not None for _,w in g.values()) else {} for g in groups]
        result=decode(t,groups,track,rows(t,'nodes',float),rows(t,'arcs',int))
        (out/(t+'.json')).write_text(json.dumps(result))
        pts=result['points']; eligible=[p for p in pts if (t,p['point_index']) in labels]
        row=dict(trip=t,points=len(pts),matched=sum(p['arc_id'] is not None for p in pts),changed=sum(p['arc_id']!=p['baseline'] for p in pts),connected_steps=sum(p['status']=='connected' for p in pts),adjacent_steps=len(pts)-1,eligible=len(eligible))
        for key in ['baseline','arc_id']:
            row[key+'_accuracy']=sum(p[key]==labels[t,p['point_index']] for p in eligible)/len(eligible) if eligible else ''
        summary.append(row);print(t,row['matched'],'/',len(pts),'matched')
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,summary[0]);writer.writeheader();writer.writerows(summary)
if __name__=='__main__':main()
