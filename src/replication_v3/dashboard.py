"""Read-only browser snapshot of verified queue state. No server/credentials."""
from __future__ import annotations
import html
from .core import machine_root,run_dir,read_json,atomic_bytes,now


def render(machine):
    from .orchestrator import get_rows,estimate
    rows=get_rows(machine)
    eta=estimate(machine)
    done=sum(r["state"]=="COMPLETED" for r in rows)
    out=[]
    for r in rows:
        progress=""
        p=run_dir(machine,r["seed"],r["condition"])/"progress.json"
        if p.exists():
            try:
                x=read_json(p)
                progress=f"{x['epoch']}/{x['total_epochs']} epochs, val {x['val_accuracy_pct']:.2f}%"
            except (ValueError,KeyError,OSError):progress="unreadable progress"
        def e(s):return html.escape(str(s),quote=True)
        out.append(f"<tr><td>{r['seed']}</td><td>{e(r['condition'])}</td>"
                   f"<td class='{e(r['state'].lower())}'>{e(r['state'])}</td>"
                   f"<td>{e(progress)}</td><td>{e(r['attempts'])}</td></tr>")
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="45"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LOFO Replication v3 - {html.escape(machine)}</title>
<style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{max-width:1000px;margin:3em auto;padding:0 1em}}
header{{display:flex;align-items:baseline;gap:1em;flex-wrap:wrap}}main{{margin-top:2em}}
progress{{width:100%;height:1.2em}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #8885;text-align:left;padding:.65em .4em;font-size:.9rem}}
.completed{{color:#238253}}.failed,.blocked{{color:#c64b46}}.running{{color:#296bad}}
small{{opacity:.75}}@media(max-width:500px){{td,th{{font-size:.7rem}}}}
</style></head><body><header><h1>LOFO Replication v3</h1><h2>{html.escape(machine)}</h2></header>
<p><strong>{done}/30 verified runs</strong> complete</p>
<progress max="30" value="{done}"></progress>
<p><small>Approximate remaining training time: {('unknown' if eta['eta_hours'] is None else str(eta['eta_hours'])+' hours')} (not including evaluation)</small></p>
<main><table><thead><tr><th>Seed</th><th>Condition</th><th>State</th><th>Progress</th><th>Attempts</th></tr></thead>
<tbody>{''.join(out)}</tbody></table></main>
<p><small>Generated {html.escape(now())} UTC. This is a static local snapshot; it is updated by the queue.</small></p>
</body></html>'''
    target=machine_root(machine)/"dashboard.html"
    atomic_bytes(target,doc.encode("utf-8"))
    return target
