"""Rebuild Figures 2-6 from immutable historical paper records.

Run: python plotting/plot_paper.py --output results/figures
Relative paths are resolved from the release root, independent of current cwd.
The renderer preserves data, line order, colors, limits and original log gaps.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/figures'
COMPACT = {f'matrix_game_rtdrm_seed_{i}' for i in range(3)} | {'kuhn_3_rtdcfr','leduc_3_rtdcfr','goofspiel_4_rtdcfr'}
GROUPS = [
    ('legend-matrix-games',[f'matrix_game_seed_{i}' for i in range(4)],5),
    ('legend-efg-games',['kuhn-3','leduc-3','goofspiel-4','liarsdice-6'],5),
    ('legend-domwu-weights',['kuhn-3-domwu','leduc-3-domwu'],4),
    ('legend-rtdrm-ablation',[f'matrix_game_rtdrm_seed_{i}' for i in range(3)],4),
    ('legend-rtdcfr-ablation',['kuhn_3_rtdcfr','leduc_3_rtdcfr','goofspiel_4_rtdcfr'],4),
]
SHA = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_data(path):
    rows=[];breaks=[]
    for text in Path(path).read_text(encoding='utf-8-sig').splitlines():
        text=text.split('#',1)[0].strip()
        if not text:breaks.append(len(rows));continue
        cols=text.split()
        try:rows.append((float(cols[0]),float(cols[2])))
        except (ValueError,IndexError):breaks.append(len(rows))
    return rows,breaks


def pdf_check(path):
    reader=PdfReader(path);page=reader.pages[0];fonts=[]
    for ref in page['/Resources'].get('/Font',{}).values():
        font=ref.get_object();desc=font['/DescendantFonts'][0].get_object() if '/DescendantFonts' in font else font
        descriptor=desc.get('/FontDescriptor');descriptor=descriptor.get_object() if descriptor else {}
        embedded=any(k in descriptor for k in ['/FontFile','/FontFile2','/FontFile3'])
        assert embedded and font['/Subtype']!='/Type3',(path,font)
        fonts.append(dict(name=str(font.get('/BaseFont')),embedded=embedded,type=str(font['/Subtype'])))
    images=sum(1 for ref in page['/Resources'].get('/XObject',{}).values() if ref.get_object().get('/Subtype')=='/Image')
    assert images==0
    return dict(sha256=SHA(path),page_points=[float(page.mediabox.width),float(page.mediabox.height)],fonts=fonts,image_objects=images)


def verify_text(fig,objects,name):
    fig.canvas.draw();r=fig.canvas.get_renderer()
    for obj in objects:
        box=obj.get_window_extent(r)
        assert box.x0>=-.5 and box.y0>=-.5 and box.x1<=fig.bbox.width+.5 and box.y1<=fig.bbox.height+.5,(name,str(obj),box)


def draw_panel(spec):
    compact=spec['output_stem'] in COMPACT
    width,height=(1.92,1.78) if compact else (2.88,2.05)
    margins=(.58,.40,.08,.07) if compact else (.60,.40,.20,.10)
    left,bottom,right,top=margins
    fig=plt.figure(figsize=(width,height));ax=fig.add_axes([left/width,bottom/height,(width-left-right)/width,(height-bottom-top)/height])
    hashes=[]
    for c in spec['curves']:
        assert SHA(c['data_path'])==c['data_sha256'],c['data_path']
        rows,breaks=read_data(c['data_path'])
        assert len(rows)==c['point_count'] and breaks==c['blank_line_break_positions']
        points=[]
        for i,(x,y) in enumerate(rows):
            if i in breaks:points.append((float('nan'),float('nan')))
            points.append((x,y if math.isfinite(y) and y>0 and math.isfinite(x) else float('nan')))
        a=np.array(points)
        ax.plot(a[:,0],a[:,1],color=c['style']['color'],linewidth=1.10,
                linestyle='-' if c['style']['source_dt']==1 else '--',solid_capstyle='butt')
        hashes.append(c['data_sha256'])
    ax.set_yscale('log',nonpositive='mask');ax.set_xlim(spec['x_range']);ax.set_ylim(spec['y_range'])
    ax.set_xlabel('Iterations',labelpad=2.4);ax.set_ylabel('Exploitability',labelpad=3.0)
    emin=round(math.log10(spec['y_range'][0]));emax=round(math.log10(spec['y_range'][1]));step=max(1,math.ceil((emax-emin)/5))
    exponents=[e for e in range(emin,emax+1) if e%step==0] or [emin,emax]
    ax.yaxis.set_major_locator(FixedLocator([10.**e for e in exponents]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:r'$10^{'+str(round(math.log10(v)))+r'}$'))
    ax.yaxis.set_minor_locator(NullLocator());ax.xaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3,steps=[1,2,2.5,5,10]))
    if compact:
        # Keep the full-size font in the narrow panel: align the endpoint text
        # inward rather than reducing its font or changing the axis range.
        ax.get_xticklabels()[-1].set_horizontalalignment('right')
    ax.tick_params(direction='out',length=2.8,pad=2.5,colors='#333333')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    for spine in ['left','bottom']:ax.spines[spine].set_color('#444444')
    ax.set_axisbelow(True);ax.grid(axis='y',which='major',color='#e9e9e9',linewidth=.45)
    verify_text(fig,[*ax.get_xticklabels(),*ax.get_yticklabels(),ax.xaxis.label,ax.yaxis.label],spec['output_stem'])
    path=OUT/(spec['output_stem']+'.pdf');fig.savefig(path);plt.close(fig)
    return dict(stem=spec['output_stem'],panel_inches=[width,height],curves=len(spec['curves']),records=sum(c['point_count'] for c in spec['curves']),
        data_hashes=hashes,x_range=spec['x_range'],y_range=spec['y_range'],legend=False,pdf=pdf_check(path))


def labels_for(group,curves):
    if group=='legend-domwu-weights':return ['All ones','Kroer (2015)','Kroer (2017)','Farina (2021)']
    if 'ablation' in group:
        pairs=[('1.5',r'-\infty'),('1.5','0'),('1.5','0.5'),('2',r'-\infty'),('2','0'),('2','0.5')]
        return [r'$(\alpha,\beta)=('+a+','+b+r')$' for a,b in pairs]+[curves[-1]['title']]
    return [c['title'].replace('+(last)','+ (last)').replace('+(avg)','+ (avg)') for c in curves]


def draw_legend(group,stems,ncol,by_stem):
    ref=by_stem[stems[0]]['curves']
    signature=lambda curves:[(c['title'],c['style']) for c in curves]
    assert all(signature(by_stem[s]['curves'])==signature(ref) for s in stems)
    labels=labels_for(group,ref)
    handles=[Line2D([],[],color=c['style']['color'],linewidth=1.4,linestyle='-' if c['style']['source_dt']==1 else '--') for c in ref]
    # Interleave input so the displayed legend reads left-to-right in source
    # method order. Ablations align alpha=1.5 and alpha=2 across the two rows.
    if len(ref)==10:order=[j for i in range(5) for j in (i,i+5)]
    elif len(ref)==7:order=[0,3,1,4,2,5,6]
    else:order=list(range(len(ref)))
    height=.46 if len(ref)>4 else .25
    fig=plt.figure(figsize=(6,height))
    legend=fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='center',ncol=ncol,
        frameon=False,fontsize=9.5,handlelength=1.35,handletextpad=.45,columnspacing=1.10,
        labelspacing=.50,borderpad=0,borderaxespad=0)
    verify_text(fig,[legend],group)
    path=OUT/(group+'.pdf');fig.savefig(path);plt.close(fig)
    return dict(stem=group,panels=stems,labels=labels,columns=ncol,display_order=order,pdf=pdf_check(path))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='Output directory, relative to release root unless absolute')
    parser.add_argument('--correct-known-labels', action='store_true',
                        help='Correct the confirmed seed3 RTDRM/RTRM+ label swap; requires a separate --output')
    args = parser.parse_args()
    if args.correct_known_labels and args.output is None:
        parser.error('--correct-known-labels requires an explicit, separate --output directory')
    output = args.output or Path('results/figures')
    OUT = (output if output.is_absolute() else ROOT/output).resolve()
    mode = 'corrected_known_labels' if args.correct_known_labels else 'historical_paper'
    if args.correct_known_labels:
        if OUT == (ROOT/'results/figures').resolve():
            parser.error('Corrected figures must use a separate output directory, such as results/figures_corrected')
        previous = OUT/'render_manifest.json'
        if previous.exists():
            old_mode = json.loads(previous.read_text(encoding='utf-8')).get('mapping_mode', 'historical_paper')
            if old_mode != mode:
                parser.error('Output contains historical figures; choose a separate directory')
        elif OUT.exists() and any(OUT.iterdir()):
            parser.error('Corrected figures require an empty output directory or an existing corrected output')
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'cmr10','font.size':9.5,'mathtext.fontset':'cm',
        'axes.formatter.use_mathtext':True,'axes.unicode_minus':False,
        'axes.labelsize':10,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
        'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
        'pdf.fonttype':42,'ps.fonttype':42,'path.simplify':False,'agg.path.chunksize':0,
        'savefig.facecolor':'white','figure.facecolor':'white'})
    source = ROOT/'plotting/paper_figures.json'
    manifest = json.loads(source.read_text(encoding='utf-8'))
    corrections = []
    if args.correct_known_labels:
        seed3 = next(spec for spec in manifest['figures'] if spec['output_stem']=='matrix_game_seed_3')
        drm = next(curve for curve in seed3['curves'] if curve['title']=='RTDRM')
        rmplus = next(curve for curve in seed3['curves'] if curve['title']=='RTRM+')
        # Keep algorithm labels/colors/order and exchange only the two datasets.
        for key in ['data_path', 'data_sha256', 'point_count', 'blank_line_break_positions']:
            drm[key], rmplus[key] = rmplus[key], drm[key]
        corrections = [{'panel':'matrix_game_seed_3',
                        'reason':'Confirmed historical RTDRM/RTRM+ label swap',
                        'data_by_label':{'RTDRM':drm['data_path'], 'RTRM+':rmplus['data_path']}}]
    for spec in manifest['figures']:
        assert spec['source_columns']==[1,3]
        for curve in spec['curves']:
            curve['data_path'] = str(ROOT/curve['data_path'])
    by_stem = {spec['output_stem']:spec for spec in manifest['figures']}
    result = dict(source_manifest_sha256=SHA(source),script_sha256=SHA(__file__),
        mapping_mode=mode,corrections=corrections,
        numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
        font='Computer Modern',panels=[],legends=[],
        numerical_policy='Original records, ordering, scales and limits; no smoothing, resampling or artificial floor',
        log_policy=manifest['log_policy'])
    for spec in manifest['figures']:
        result['panels'].append(draw_panel(spec))
        print('panel',spec['output_stem'],flush=True)
    for group,stems,ncol in GROUPS:
        result['legends'].append(draw_legend(group,stems,ncol,by_stem))
        print('legend',group,flush=True)
    assert sum(spec['curves'] for spec in result['panels'])==130
    assert sum(spec['records'] for spec in result['panels'])==13226
    (OUT/'render_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
