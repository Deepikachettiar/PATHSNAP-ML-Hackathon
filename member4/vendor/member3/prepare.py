"""Optional regeneration: python prepare.py UPLOAD_FOLDER (numpy, pandas, joblib, sklearn)."""
import gzip,io,zipfile,sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
base=Path(sys.argv[1])
z=zipfile.ZipFile(base/'member2.zip')
bundle=joblib.load(io.BytesIO(z.read('results_zip\\models\\extra_trees.joblib')))
original=pd.read_csv(io.BytesIO(z.read('results_zip\\student_scores_extra_trees.csv.gz')),compression='gzip',dtype={'trip_id':str})
df=pd.read_csv(base/'candidates.csv',compression='zip',dtype={'trip_id':str})
df=df[df.split=='student_test'].copy()
df.to_csv('data/candidates.gz',compression='gzip',index=False)
labels=[];scores=[]
for t,g in df.groupby('trip_id'):
    def read(ext,dtype=float):
        return np.loadtxt('data/'+t+'.'+ext,dtype=dtype)
    track=read('track');nodes=read('nodes');arcs=read('arcs',int)
    scale=np.array([111195*np.cos(np.deg2rad(track[:,1].mean())),111195])
    xy=nodes*scale;gps=track[:,:2]*scale
    start,end=xy[arcs[:,0]],xy[arcs[:,1]]
    direction=end-start
    length=np.linalg.norm(direction,axis=1)
    arc=g.arc_id.to_numpy();idx=g.point_index.to_numpy()
    snap=g[['projected_lon','projected_lat']].to_numpy()*scale
    frac=np.clip(np.sum((snap-start[arc])*direction[arc],axis=1)/np.maximum(length[arc]**2,1e-12),0,1)
    delta=np.diff(gps,axis=0,prepend=gps[:1])
    heading=lambda v: np.degrees(np.arctan2(v[:,0],v[:,1]))%360
    g=g.copy()
    g['arc_length_m']=length[arc];g['arc_fraction']=frac
    g['distance_from_arc_start_m']=frac*length[arc];g['distance_to_arc_end_m']=(1-frac)*length[arc]
    g['arc_bearing_deg']=heading(direction[arc]);g['gps_heading_deg']=heading(delta)[idx]
    g['heading_difference_deg']=abs((g.arc_bearing_deg-g.gps_heading_deg+180)%360-180)
    g['dt_s']=np.diff(track[:,2],prepend=track[0,2])[idx]
    g['gps_step_m']=np.linalg.norm(delta,axis=1)[idx]
    g['gps_speed_m_s']=g.gps_step_m/g.dt_s.replace(0,np.nan)
    g['from_out_degree']=np.bincount(arcs[:,0])[arcs[arc,0]]
    g['to_in_degree']=np.bincount(arcs[:,1])[arcs[arc,1]]
    X=g[bundle['features']]
    model=bundle['model']
    features=model.steps[0][1].transform(X)
    forest=model.steps[-1][1]
    prob=sum(tree.predict_proba(features)[:,1] for tree in forest.estimators_)/len(forest.estimators_)
    s=g[['trip_id','point_index','arc_id']].copy()
    s['candidate_probability']=np.maximum(prob,1e-6)
    s.candidate_probability/=s.groupby('point_index').candidate_probability.transform('sum')
    keys=['trip_id','point_index','arc_id']
    s=s.merge(original[keys+['candidate_probability']],on=keys,how='left',validate='one_to_one',suffixes=('','_original'))
    s['candidate_probability']=s.pop('candidate_probability_original').fillna(s.candidate_probability)
    scores.append(s)
    # Monotone minimum-total GPS-to-reference distance. Staying and forward skips allowed.
    route=read('route',int);v=direction[route];origin=start[route];den=np.maximum(length[route]**2,1e-12)
    prior=np.zeros(len(route));back=[]
    for point in gps:
        d=point-origin
        cost=np.linalg.norm(d-np.clip((d*v).sum(1)/den,0,1)[:,None]*v,axis=1)
        mins=np.minimum.accumulate(prior)
        arg=np.maximum.accumulate(np.where(prior==mins,np.arange(len(route)),0))
        back.append(arg.astype('int32'));prior=cost+mins
    j=int(prior.argmin());answer=[]
    for arg in reversed(back):
        answer.append(route[j]);j=arg[j]
    labels.extend((t,i,a) for i,a in enumerate(reversed(answer)))
pd.concat(scores).to_csv('data/scores.gz',compression='gzip',index=False)
pd.DataFrame(labels,columns=['trip_id','point_index','reference_arc_id']).to_csv('data/derived_labels.csv',index=False)
