"""Build a standalone report from the current decoder outputs."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def chart(title,subtitle,rows,scale,unit):
    parts=[f'<section><h2>{title}</h2><p>{subtitle}</p><p><b>Blue</b> Extra Trees &nbsp; <b>Green</b> HMM + Viterbi</p>']
    for trip,a,b in rows:
        parts.append(f'<div class="row"><code>{trip}</code><div>')
        for value,color in [(a,'#33689b'),(b,'#097968')]:
            parts.append(f'<div class="bar"><i style="width:{100*value/scale:.3f}%;background:{color}"></i><span>{value:.2f}{unit}</span></div>')
        parts.append('</div></div>')
    return ''.join(parts)+'</section>'

def build():
    out=ROOT/'outputs/final'
    with (out/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    with (out/'sequence_metrics.csv').open() as f:seq={(r['trip'],r['method']):float(r['edits_per_reference_entry']) for r in csv.DictReader(f)}
    accuracy=[(r['trip'],100*float(r['baseline_accuracy']),100*float(r['arc_id_accuracy'])) for r in summary]
    edits=[(t,seq[t,'baseline'],seq[t,'arc_id']) for t,a,b in accuracy]
    html=(ROOT/'view.html').read_text()
    data={p.stem:json.loads(p.read_text()) for p in sorted(out.glob('*.json'))}
    setup="""const bundled=DATA;
function selectTrip(){route=bundled[trips.value];slider.max=Math.max(0,route.replay.length-1);slider.value=0;draw()}
for(const id of Object.keys(bundled).sort()){trips.add(new Option(id,id))}
trips.onchange=selectTrip;selectTrip();""".replace('=DATA;','='+json.dumps(data,separators=(',',':'))+';')
    html=html.replace('</script>',setup+'</script>')
    html+='''<style>section{margin:3em 0}.row{display:grid;grid-template-columns:95px 1fr;align-items:center;margin:10px 0}.bar{display:flex;align-items:center;gap:8px;padding-right:65px;height:20px}.bar i{height:10px;flex-shrink:0}.bar span{font:13px system-ui}select{padding:6px}</style>'''
    html+=chart('Road accuracy by trip','Share of GPS observations assigned to the derived reference road. Higher is better.',accuracy,100,'%')
    html+=chart('Route sequence error by trip','Edits per reference road entry after collapsing repeated predicted roads. Lower is better.',edits,max(max(a,b) for t,a,b in edits) or 1,'')
    (ROOT/'Member3_Dashboard.html').write_text(html,encoding='utf-8')
if __name__=='__main__':build()
