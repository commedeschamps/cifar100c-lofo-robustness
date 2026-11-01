"""Regenerate manuscript assets from immutable saved condition counts; no ML imports.

Run from any directory with the recorded matplotlib/NumPy environment:
    MPLCONFIGDIR=/tmp/research_cv_mpl .venv/bin/python paper/generate_assets.py
The compact submission's essential_results are also supported automatically.
"""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import csv
import hashlib
import json
import math
import os
import argparse
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP

os.environ.setdefault("MPLCONFIGDIR", "/tmp/research_cv_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
FAMILIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
COLORS = dict(noise="#0F4D92", blur="#42949E", weather="#B64342", digital="#9A4D8E")
PATHS = {
    "historical": ROOT / "results/evaluation/seed42/metrics/evaluation.json",
    "controlled_v2": ROOT / "results/evaluation_controlled_v2/seed42/metrics/evaluation.json",
}
if not PATHS["historical"].is_file():
    PATHS = {cohort: ROOT / "reproducibility/essential_results" / cohort / "metrics/evaluation.json"
             for cohort in ("historical", "controlled_v2")}
MODELS = [
    ("historical", "clean_baseline", "Historical clean"),
    ("controlled_v2", "clean_baseline", "Primary clean"),
    *(("historical", "lofo_" + f, "LOFO-" + f.title()) for f in FAMILIES),
    ("controlled_v2", "all_four_representatives", "All-four control"),
]


def display(value, places=2, signed=False):
    """Round the exact count-derived rational, avoiding binary-float decimal ties."""
    exact = Fraction(float(value)).limit_denominator(7500)
    assert abs(float(exact)-float(value)) < 1e-10
    number = Decimal(exact.numerator) / Decimal(exact.denominator)
    number = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return format(number, ('+' if signed else '')+'.'+str(places)+'f')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(only_figure=None):
    (PAPER / "figures").mkdir(exist_ok=True)
    rows, inputs = {}, {}
    for cohort, path in PATHS.items():
        inputs[str(path.relative_to(ROOT))] = sha(path)
        records = json.loads(path.read_text())["conditions"]
        for r in records:
            key = cohort, r["model"], r["corruption"], r["severity"]
            assert key not in rows and r["n_samples"] == 10000 and r["seed"] == 42
            assert math.isclose(r["accuracy_pct"], 100*r["n_correct"]/r["n_samples"], abs_tol=1e-10)
            rows[key] = r
    assert len(rows) == 532

    def accuracy(cohort, model, types, severities=range(1, 6)):
        return mean(100*rows[cohort, model, c, s]["n_correct"]/10000
                    for c in types for s in severities)

    all_types = sum(FAMILIES.values(), [])
    with (PAPER / "tables/seed42_results.csv").open() as stream:
        existing = {(r["cohort"], r["model"]): r for r in csv.DictReader(stream)}
    inputs["paper/tables/seed42_results.csv"] = sha(PAPER / "tables/seed42_results.csv")
    overall = []
    for cohort, model, label in MODELS:
        clean = rows[cohort, model, "clean", 0]["accuracy_pct"]
        corr = accuracy(cohort, model, all_types)
        old = existing[cohort, model]
        assert math.isclose(corr, float(old["overall_pct"]), abs_tol=1e-10)
        assert clean == float(old["clean_pct"])
        overall.append(dict(cohort=cohort, model=model, label=label, clean_pct=clean,
                            overall_pct=corr, best_epoch=int(old["best_epoch"]),
                            family_accuracy={f: accuracy(cohort, model, cs) for f, cs in FAMILIES.items()}))
    family_rows, type_rows, severity_rows, cells = [], [], [], []
    for f, cs in FAMILIES.items():
        def base(types=cs, severities=range(1, 6)):
            return accuracy("controlled_v2", "clean_baseline", types, severities)
        def lofo(types=cs, severities=range(1, 6)):
            return accuracy("historical", "lofo_"+f, types, severities)
        def control(types=cs, severities=range(1, 6)):
            return accuracy("controlled_v2", "all_four_representatives", types, severities)
        b, l, a = base(), lofo(), control()
        family_rows.append(dict(family=f, baseline_pct=b, lofo_pct=l, gain_pp=l-b,
                                all_four_pct=a, all_four_minus_lofo_pp=a-l))
        for c in cs:
            b, l = base([c]), lofo([c])
            type_rows.append(dict(family=f, corruption=c, baseline_pct=b, lofo_pct=l, gain_pp=l-b))
            for s in range(1, 6):
                b, l = base([c], [s]), lofo([c], [s])
                cells.append(dict(family=f, corruption=c, severity=s, baseline_pct=b, lofo_pct=l, gain_pp=l-b))
        for s in range(1, 6):
            b, l, a = base(severities=[s]), lofo(severities=[s]), control(severities=[s])
            severity_rows.append(dict(family=f, severity=s, baseline_pct=b, lofo_pct=l,
                                      all_four_pct=a, gain_pp=l-b))

    def table(filename, caption, label, columns, header, body):
        out = ("% Generated by paper/generate_assets.py from saved counts.\n"
               "\\begin{table}[htbp]\n\\caption{"+caption+"}\n\\label{"+label+"}\n"
               "\\centering\n\\setlength{\\tabcolsep}{8pt}\n"
               "\\begin{tabular}{"+columns+"}\n\\toprule\n"+header+" \\\\\n\\midrule\n"
               +"\n".join(body)+"\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        (PAPER/"tables"/filename).write_text(out)

    table("conditions.tex", "Six principal conditions. G, B, F, and J denote Gaussian noise, defocus blur, fog, and JPEG compression. One operator is sampled per corrupted presentation.",
          "tab:conditions", "lcl", r"Condition & $p$ & Allowed representatives",
          [r"Primary clean & 0 & None \\", r"LOFO-Noise & 0.5 & B, F, J \\",
           r"LOFO-Blur & 0.5 & G, F, J \\", r"LOFO-Weather & 0.5 & G, B, J \\",
           r"LOFO-Digital & 0.5 & G, B, F \\", r"All-four control & 0.5 & G, B, F, J \\"])
    labels = ["Historical clean$^{h}$", "Primary clean$^{v}$", "LOFO-Noise", "LOFO-Blur", "LOFO-Weather", "LOFO-Digital", "All-four control$^{v}$"]
    table("overall_performance.tex", "Seven completed seed-42 runs. Accuracy (\\%); corrupted accuracy averages 75 conditions. Epoch is selected by clean validation. $h$: historical clean reference; $v$: controlled-v2; other rows: LOFO. Recorded cohorts are detailed in Appendix~\\ref{app:records}.",
          "tab:overall", "lrrr", r"Model & Epoch & Clean & Corrupted",
          [f"{lab} & {r['best_epoch']} & {r['clean_pct']:.2f} & {r['overall_pct']:.2f} \\\\" for lab, r in zip(labels, overall)])
    table("held_out_transfer.tex", "Held-out-family accuracy (\\%) and gain (pp) against the primary MPS/FP32 clean baseline. Each row uses a different LOFO network; the all-four model has explicit family exposure. Differences use unrounded measurements.",
          "tab:transfer", "lrrrr", r"Family & Clean & LOFO & $\Delta$ (pp) & All-four",
          [f"{r['family'].title()} & {r['baseline_pct']:.2f} & {r['lofo_pct']:.2f} & {r['gain_pp']:+.2f} & {r['all_four_pct']:.2f} \\\\" for r in family_rows])

    for name, records in [("held_out_transfer.csv",family_rows),("per_corruption_transfer.csv",type_rows),
                          ("severity_transfer.csv",severity_rows),("held_out_conditions.csv",cells)]:
        with (PAPER/"tables"/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
                         "font.size":8, "axes.labelsize":8, "axes.titlesize":8,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "axes.linewidth":0.65, "lines.linewidth":1.3, "legend.frameon":False,
                         "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none",
                         "svg.hashsalt":"research_cv_cifar100_lofo"})
    def save(fig, name):
        if only_figure is not None and name != only_figure:
            plt.close(fig)
            return
        for ext in ["pdf","svg","png"]:
            metadata = {"CreationDate":None,"ModDate":None} if ext=="pdf" else {"Date":None} if ext=="svg" else {"Software":"research_cv manuscript assets"}
            fig.savefig(PAPER/"figures"/(name+"."+ext),dpi=300,bbox_inches="tight",pad_inches=0.025,metadata=metadata)
        plt.close(fig)

    # Schematic: example is explicitly one of four separately trained LOFO models.
    fig, ax=plt.subplots(figsize=(6.4,3.0)); ax.set_xlim(0,16); ax.set_ylim(0,8); ax.axis("off")
    def box(x,y,w,h,text,color="#EAF0F6",edge="#0F4D92",dashed=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.08",facecolor=color,
                                   edgecolor=edge,linewidth=.8,linestyle="--" if dashed else "-"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    box(.2,5.8,3.2,1.1,"Noise: Gaussian\nEXCLUDED",color="#F6CFCB",edge="#B64342",dashed=True)
    for x,t in [(4.3,"Blur\ndefocus blur"),(8.4,"Weather\nfog"),(12.5,"Digital\nJPEG")]:
        box(x,5.8,3.2,1.1,t)
        ax.annotate("",xy=(6.5,4.4),xytext=(x+1.6,5.65),arrowprops=dict(arrowstyle="->",lw=.8,color="#0F4D92"))
    box(.2,2.9,3.0,1.3,"Training image\nCrop + flip")
    box(4.3,2.7,4.4,1.7,"50% clean\n50%: one operator\nSeverity 1–5")
    box(10.0,2.9,5.4,1.3,"Normalize\nScratch ResNet-18")
    for start,end in [((3.35,3.55),(4.15,3.55)),((8.85,3.55),(9.85,3.55))]:
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",lw=.9))
    box(10.0,.3,5.4,1.3,"Held-out Noise evaluation\n3 types × 5 severities",color="#FFF7F5",edge="#B64342")
    ax.annotate("",xy=(12.7,1.75),xytext=(12.7,2.75),arrowprops=dict(arrowstyle="->",lw=.9))
    ax.text(8,7.8,"Example: LOFO-Noise",ha="center",va="top",fontweight="bold",fontsize=11)
    ax.text(.2,1.0,"Four separately trained LOFO models\nFull evaluation: 75 conditions per model",ha="left",va="center",fontsize=10)
    save(fig,"lofo_design")

    fig,ax=plt.subplots(figsize=(3.45,2.5))
    for r in overall:
        model=r['model']; cohort=r['cohort']; x=r['clean_pct']; y=r['overall_pct']
        if model.startswith('lofo_'):
            f=model[5:]; col=COLORS[f]; marker='o'; lab=f"LOFO-{f.title()}"; offset={'noise':(6,-10),'blur':(6,-3),'weather':(5,-12),'digital':(6,-3)}[f]
        elif model=='all_four_representatives': col='#272727'; marker='s'; lab='All-four (v2)'; offset=(5,6)
        elif cohort=='historical': col='#767676'; marker='x'; lab='Historical clean'; offset=(-70,5)
        else: col='#272727'; marker='D'; lab='Primary clean (v2)'; offset=(-67,-13)
        ax.scatter(x,y,c=col,marker=marker,s=27,zorder=3)
        ax.annotate(lab,(x,y),xytext=offset,textcoords='offset points',fontsize=6.8,color=col)
    ax.set(xlabel='Clean test accuracy (%)',ylabel='Mean corrupted accuracy (%)',xlim=(70.4,77.5),ylim=(44,69.5))
    ax.set_xticks([71,73,75,77]); ax.set_yticks([45,50,55,60,65]); ax.grid(axis='y',alpha=.16,lw=.5)
    fig.tight_layout(pad=.6); save(fig,"clean_vs_corrupted")

    fig,ax=plt.subplots(figsize=(3.45,2.0))
    gains=[r['gain_pp'] for r in family_rows]
    bars=ax.bar(range(4),gains,color=[COLORS[f] for f in FAMILIES],edgecolor='#272727',lw=.6,width=.62)
    for bar,v in zip(bars,gains): ax.text(bar.get_x()+bar.get_width()/2,v+.4,f'+{v:.2f}',ha='center',fontsize=8)
    ax.set_xticks(range(4),[f.title() for f in FAMILIES]); ax.set(ylabel='Held-out gain (pp)',ylim=(0,22)); ax.set_yticks([0,5,10,15,20]); ax.axhline(0,color='#272727',lw=.7)
    fig.tight_layout(pad=.6); save(fig,"family_transfer")

    # Same row model is used at every severity and mean in that corruption's family.
    matrix=np.array([[next(r['gain_pp'] for r in cells if r['corruption']==t['corruption'] and r['severity']==s)
                      for s in range(1,6)]+[t['gain_pp']] for t in type_rows])
    fig,ax=plt.subplots(figsize=(3.45,4.05))
    im=ax.imshow(matrix,cmap='RdBu',norm=TwoSlopeNorm(vmin=-40,vcenter=0,vmax=40),aspect='auto')
    names=[r['corruption'].replace('_',' ').replace('jpeg','JPEG') for r in type_rows]
    ax.set_yticks(range(15),names,fontsize=7); ax.set_xticks(range(6),['1','2','3','4','5','Mean']); ax.xaxis.tick_top()
    ax.tick_params(axis='both',length=0,pad=3)
    for i in range(15):
        for j in range(6):
            v=matrix[i,j]; ax.text(j,i,display(v,1,signed=True),ha='center',va='center',fontsize=6.2,color='white' if abs(v)>25 else '#202020')
    for y in [2.5,6.5,10.5]: ax.axhline(y,color='white',lw=2)
    ax.axvline(4.5,color='#272727',lw=.7)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.035,aspect=25)
    cb.set_label('LOFO − primary clean (pp)',fontsize=7); cb.ax.tick_params(labelsize=7,length=2)
    fig.tight_layout(pad=.5); save(fig,"corruption_transfer_heatmap")

    fig,axes=plt.subplots(2,2,figsize=(3.45,3.0),sharex=True,sharey=True)
    for ax,(f,cs) in zip(axes.flat,FAMILIES.items()):
        records=[r for r in severity_rows if r['family']==f]
        for key,col,sty,marker in [('baseline_pct','#767676','--','.'),('lofo_pct',COLORS[f],'-','o'),('all_four_pct','#272727',':','s')]:
            ax.plot(range(1,6),[r[key] for r in records],color=col,ls=sty,marker=marker,ms=2.8,lw=1)
        ax.set_title(f.title(),pad=3); ax.set_xticks(range(1,6)); ax.set_ylim(0,80); ax.set_yticks([0,20,40,60,80]); ax.tick_params(labelsize=7)
        ax.grid(axis='y',alpha=.15,lw=.5)
    fig.supxlabel('Severity',fontsize=8,y=.09); fig.supylabel('Family accuracy (%)',fontsize=8,x=.01)
    handles=[plt.Line2D([],[],c='#767676',ls='--',label='Primary clean'),plt.Line2D([],[],c='#0F4D92',marker='o',ms=3,label='Held-out LOFO'),plt.Line2D([],[],c='#272727',ls=':',label='All-four')]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=6.5,columnspacing=.9,handlelength=1.5,bbox_to_anchor=(.51,-.015))
    fig.tight_layout(rect=(.035,.12,1,1),pad=.6,w_pad=.7,h_pad=.7); save(fig,"severity_transfer")

    metrics={"overall":overall,"held_out_families":family_rows,"per_corruption":type_rows,
             "severity":severity_rows,"held_out_conditions":cells,
             "negative_held_out_cells":sum(r['gain_pp']<0 for r in cells),
             "zero_held_out_cells":sum(r['gain_pp']==0 for r in cells),
             "blur_gain_without_glass_pp":mean(r['gain_pp'] for r in type_rows if r['family']=='blur' and r['corruption']!='glass_blur')}
    (PAPER/"tables/plot_data.json").write_text(json.dumps(metrics,indent=2)+"\n")
    outputs=[p for p in (PAPER/"figures").iterdir() if p.is_file()]
    outputs += [PAPER/"tables"/n for n in ['conditions.tex','overall_performance.tex','held_out_transfer.tex',
                 'held_out_transfer.csv','per_corruption_transfer.csv','severity_transfer.csv','held_out_conditions.csv','plot_data.json']]
    provenance={"method":"Recomputed arithmetic from saved integer correct counts; no training or inference; no resampling.",
                "heatmap_display_rounding":"Exact count-derived rational reconstructed with denominator <=7500; decimal ROUND_HALF_UP at one place. Raw metrics unchanged.",
                "primary_baseline":"controlled_v2/clean_baseline", "inputs":inputs,
                "script_sha256":sha(Path(__file__)),"outputs":{str(p.relative_to(ROOT)):sha(p) for p in sorted(outputs)}}
    (PAPER/"tables/manuscript_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps({"conditions":len(rows),"models":len(overall),"families":family_rows,
                      "negative_held_out_cells":metrics['negative_held_out_cells'],
                      "blur_gain_without_glass_pp":metrics['blur_gain_without_glass_pp']},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--only-figure', choices=['lofo_design', 'clean_vs_corrupted',
                           'family_transfer', 'corruption_transfer_heatmap', 'severity_transfer'])
    selection.add_argument('--only-heatmap', action='store_true', help='Compatibility alias.')
    args = parser.parse_args()
    build('corruption_transfer_heatmap' if args.only_heatmap else args.only_figure)
