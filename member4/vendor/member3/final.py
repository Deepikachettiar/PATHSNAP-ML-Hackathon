from run import main,ROOT
from evaluate import main as evaluate
from dashboard import build
import csv
if __name__=='__main__':
    main()
    evaluate('final')
    build()
    with (ROOT/'outputs/final/summary.csv').open() as f:
        rows=list(csv.DictReader(f))
    count=sum(int(r['eligible']) for r in rows)
    for key in ['baseline_accuracy','arc_id_accuracy']:
        correct=sum(round(float(r[key])*int(r['eligible'])) for r in rows)
        print(key,correct,'/',count,format(correct/count,'.2%'),'against derived reference labels')
    print('Open Member3_Dashboard.html for the replay and graphs.')
