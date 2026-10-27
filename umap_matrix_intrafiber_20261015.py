"""Frozen UMAP export and separately scoped, provisional intrafiber cell counts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import time

ROOT = Path('/Users/lownikj/ucla_dmd')
REL = Path('Primary Analyses/20261015_UMAP_matrix_and_intrafiber_cells_v1')
OUT = ROOT / REL
ID = ROOT / 'Primary Analyses/20260924_collaborator_methods_figures_v1/cell_identity/tables'
UMAP = ROOT / 'Primary Analyses/20261009_cell_type_UMAP_annotated_only_v1/Source_Data/UMAP_annotated_plotted_cells.parquet'
BACK = ROOT / 'exports/dmd_collaborator_mechanism_analysis_20260904/v1/tables/cell_marker_backbone.parquet'
INTEGRATION = ROOT / 'exports/review_informed_myofiber_marker_cell_integration_v10_completed_20260828/v1/tables'
H5 = Path('/Volumes/WALLE/ucla_dmd_cell_typing_20260813/ucla_dmd_cells_basic_lineage_typed.h5ad')
DMDROOT = Path('/Volumes/WALLE/ucla_dmd_review_informed_myofiber_reanalysis_20260827/dmd_masks_v1')
NORMAL = Path('/Volumes/WALLE/ucla_dmd_myofiber_integrated_analysis_20260817/normal_morphometry/tables/primary_myofiber_manifest.parquet')
THRESH = ROOT / 'Primary Analyses/20260922_v1/cd8/tables/frozen_thresholds.csv'
PIX = .28
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib_cell_type_umap_20261009'))

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {'CD4 T-like':'#4477AA','CD8 T-like':'#225588','NK-like':'#66AADD',
          'APC-like immune':'#AA3377','Other immune-like':'#CC99BB',
          'Endothelial-like':'#228833','Mural/perivascular-like':'#66AA55',
          'Fibroblast-like':'#CCBB44','PAX7-high myogenic candidate':'#EE7733'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def jsave(p, data):
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def savefig(fig, stem):
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT / f'Figures/{ext.upper()}/{stem}.{ext}', dpi=600,
                    bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / f'validation/{stem}_preview.png', dpi=140,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def tables(resume=False):
    if OUT.exists() and not resume:
        raise FileExistsError(f'Preserving existing result: {OUT}')
    for folder in ['UMAP','Intra_myofiber','Figures/PNG','Figures/SVG','Figures/PDF',
                   'validation','reproducibility']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    print('Joining frozen IDs, marker values, and the existing annotated-only embedding.', flush=True)
    u = pd.read_parquet(UMAP)
    b = pd.read_parquet(BACK)
    s = pd.read_parquet(ID / 'cell_marker_source_signed_means.parquet')
    r = pd.read_parquet(ID / 'cell_annotation_rank_source.parquet').set_index('cell_id').loc[b.cell_id].reset_index()
    assert len(u)==75664 and len(b)==len(s)==len(r)==205675
    assert u.cell_id.is_unique and b.cell_id.is_unique and s.cell_id.is_unique and r.cell_id.is_unique
    for k in ['cell_id','donor_id','slide_id','piece_id','lineage_cell_type']:
        assert np.array_equal(b[k].astype(str),s[k].astype(str))
        assert np.array_equal(b[k].astype(str),r[k].astype(str))
    assert set(u.cell_id)==set(s.loc[~s.lineage_cell_type.astype(str).eq('Unresolved/other'),'cell_id'])
    a = ad.read_h5ad(H5, backed='r')
    obs = a.obs.copy(); obs.index = obs.cell_id.astype(str)
    positions = obs.index.get_indexer(s.cell_id)
    assert (positions>=0).all() and obs.index.is_unique
    assert np.array_equal(obs.iloc[positions].lineage_cell_type.astype(str),s.lineage_cell_type.astype(str))
    # Read a 277,441 x 48 float32 matrix, then select the 205,675 frozen retained cells.
    values = np.asarray(a.X[:], dtype=np.float32)[positions]
    names = a.var_names.astype(str).tolist()
    assert values.shape==(205675,48) and np.isfinite(values).all()
    for marker in [k for k in s if k in names]:
        np.testing.assert_array_equal(values[:,names.index(marker)],s[marker].to_numpy())
    markercols = ['marker_'+re.sub(r'[^A-Za-z0-9_]+','_',m).strip('_') for m in names]
    assert len(set(markercols))==48
    dictionary = a.var.copy().reset_index(names='source_marker')
    dictionary.insert(1,'csv_column',markercols)
    dictionary['values']='Signed paired-background-subtracted cell-mask mean (M0); no later slide offsets or clipping'
    dictionary.to_csv(OUT / 'UMAP/Marker_dictionary.csv', index=False)
    o = obs.iloc[positions].copy().reset_index(drop=True)
    a.file.close()
    meta = s[['cell_id','donor_id','clinical_status','slide_id','piece_id','lineage_cell_type']].copy()
    meta['donor_id']=meta.donor_id.astype(str)
    meta['group']=meta.clinical_status.astype(str).replace({'Normal':'Healthy'})
    full = meta.merge(u[['cell_id','severity','UMAP1','UMAP2']],on='cell_id',how='left',validate='one_to_one',sort=False)
    full['severity']=full.donor_id.map(u[['donor_id','severity']].drop_duplicates().set_index('donor_id').severity)
    full['included_in_UMAP']=full.UMAP1.notna()
    full[['centroid_x_um','centroid_y_um']]=b[['centroid_x_um','centroid_y_um']].to_numpy()
    full[markercols]=values
    plotted = u[['cell_id']].merge(full,on='cell_id',validate='one_to_one',sort=False)
    np.testing.assert_array_equal(plotted[['UMAP1','UMAP2']],u[['UMAP1','UMAP2']])
    plotted.to_csv(OUT / 'UMAP/UMAP_cells_marker_expression_coordinates.csv',index=False)
    full.to_csv(OUT / 'UMAP/All_retained_cells_marker_expression.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    full.to_parquet(OUT / 'UMAP/All_retained_cells_marker_expression.parquet',index=False)
    counts = pd.crosstab(u.lineage_cell_type,u.group).reindex(index=list(COLORS),columns=['Healthy','DMD'],fill_value=0)
    counts['Total_N']=counts.sum(axis=1)
    counts['Donors_present']=u.groupby('lineage_cell_type',observed=True).donor_id.nunique().reindex(counts.index)
    counts.reset_index().to_csv(OUT / 'UMAP/UMAP_category_counts.csv',index=False)
    grid=u[['donor_id','group','severity']].drop_duplicates().merge(pd.DataFrame({'lineage_cell_type':list(COLORS)}),how='cross')
    dc=u.groupby(['donor_id','lineage_cell_type'],observed=True).size().rename('cell_N').reset_index()
    grid=grid.merge(dc,on=['donor_id','lineage_cell_type'],how='left',validate='one_to_one');grid.cell_N=grid.cell_N.fillna(0).astype(int)
    grid.to_csv(OUT / 'UMAP/UMAP_category_counts_by_donor.csv',index=False)
    assert counts.Total_N.sum()==75664 and grid.cell_N.sum()==75664
    plot_umap(u,counts)

    cells = meta.copy()
    for marker,key in [('CD45_555','CD45'),('CD4_750','CD4'),('CD8a_555','CD8'),('CD80','CD80'),('Granzyme K_647','GZMK'),('Granzyme B','GZMB')]:
        cells[key+'_signed_mean']=s[marker].to_numpy()
        cells[key+'_original_annotation_rank']=r[marker].to_numpy()
    for col in ['centroid_x_um','centroid_y_um','canonical_fiber_id','inside_final_fiber','signed_boundary_distance_um','boundary_distance_um']:
        cells[col]=b[col].to_numpy()
    for marker,cut in [('CD45',.90),('CD4',.90),('CD8',.85),('CD80',.90)]:
        cells[marker+'_high']=(cells[marker+'_original_annotation_rank']>=cut)&(cells[marker+'_signed_mean']>0)
    threshold = pd.read_csv(THRESH).set_index('slide_id')
    for key,col in [('GZMK','k_threshold_q99'),('GZMB','b_threshold_q99')]:
        cells[key+'_asinh_threshold']=cells.slide_id.map(threshold[col]).astype(float)
        cells[key+'_positive']=np.arcsinh(cells[key+'_signed_mean']/5)>=cells[key+'_asinh_threshold']
    cells['CD45_CD4_high']=cells.CD45_high&cells.CD4_high
    cells['CD45_CD8_high']=cells.CD45_high&cells.CD8_high
    for col in ['tile_id','local_label','cell_area_px','bbox_x0_px','bbox_x1_px','bbox_y0_px','bbox_y1_px','centroid_x_px','centroid_y_px','touches_crop_edge','touches_slide_edge']:
        cells[col]=o[col].to_numpy()
    cells['severity']=cells.donor_id.map(u[['donor_id','severity']].drop_duplicates().set_index('donor_id').severity)
    assert cells.severity.notna().all()
    cells.to_parquet(OUT / 'reproducibility/cell_gate_and_geometry_inputs.parquet',index=False)
    rules=[{'marker':m,'rule':f'original pooled PC1-adjusted annotation rank >= {v}; signed mean > 0',
            'interpretation':'Relative marker-high; not a calibrated absolute positivity threshold'}
           for m,v in [('CD45',.9),('CD4',.9),('CD8',.85),('CD80',.9)]]
    pd.DataFrame(rules).to_csv(OUT / 'Intra_myofiber/Marker_gate_rules.csv',index=False)
    threshold.reset_index().to_csv(OUT / 'Intra_myofiber/Frozen_granzyme_thresholds.csv',index=False)
    jsave(OUT / 'validation/export_checks.json', {'status':'passed','UMAP_cells':75664,
          'retained_cells':205675,'markers':48,'donors':18,'unresolved_not_plotted':130011,
          'UMAP_coordinates_unchanged_exactly':True,'one_to_one_ID_joins':True,
          '18_original_marker_columns_exactly_match_frozen_source':True,
          'additional_slide_correction':False,'negative_signal_values_retained':True})
    jsave(OUT / 'validation/source_provenance.json',{'files':[{'path':str(p),'sha256':sha(p)} for p in
          [UMAP,BACK,ID/'cell_marker_source_signed_means.parquet',ID/'cell_annotation_rank_source.parquet',H5,THRESH,INTEGRATION/'final_fiber_manifest.parquet']],
          'spatial_scope':'All 205,675 retained cells, not only cells plotted in UMAP',
          'matrix_scope':'All 48 acquired marker channels, including intracellular markers; not surface-protein-specific measurements'})
    print(counts.to_string(),flush=True)
    containment(cells)


def plot_umap(u, counts):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(10,7))
    v=u.iloc[np.random.default_rng(20261009).permutation(len(u))]
    ax.scatter(v.UMAP1,v.UMAP2,c=v.lineage_cell_type.astype(str).map(COLORS),s=1.8,alpha=.65,linewidths=0,rasterized=True)
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=6,markerfacecolor=c,markeredgewidth=0,
                   label=f'{k} (N={int(counts.loc[k,"Total_N"]):,})') for k,c in COLORS.items()]
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.03,1),frameon=False,fontsize=10,labelspacing=1.05)
    ax.set_title('COMET cell types',loc='left',fontsize=13,pad=14)
    for key,setlim in [('UMAP1',ax.set_xlim),('UMAP2',ax.set_ylim)]:
        lo,hi=u[key].min(),u[key].max();pad=(hi-lo)*.035;setlim(lo-pad,hi+pad)
    ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2');ax.set_xticks([]);ax.set_yticks([])
    for side in ['top','right']:ax.spines[side].set_visible(False)
    ax.set_aspect('equal',adjustable='box');fig.subplots_adjust(left=.08,bottom=.1,right=.65,top=.91)
    savefig(fig,'UMAP_cell_types_with_cell_N')


def cell_pixels(row, tiles, arrays):
    tile=tiles[row.slide_id].loc[int(row.tile_id)]
    key=(str(row.slide_id),int(row.tile_id))
    if key not in arrays:
        # Keep mmap references, not copies of full cell-label arrays.
        arrays[key]=np.load(tile.raw_labels_path,mmap_mode='r')
    x0,x1,y0,y1=[int(getattr(row,k)) for k in ['bbox_x0_px','bbox_x1_px','bbox_y0_px','bbox_y1_px']]
    z=arrays[key][y0-int(tile.crop_y0):y1-int(tile.crop_y0),x0-int(tile.crop_x0):x1-int(tile.crop_x0)]
    yy,xx=np.nonzero(z==int(row.local_label))
    assert len(yy)==int(row.cell_area_px),(row.cell_id,len(yy),row.cell_area_px)
    return yy+y0,xx+x0


def containment(cells):
    print('Reopening native cell masks for intrafiber candidates.',flush=True)
    needed=cells.CD45_CD4_high|cells.CD45_CD8_high|cells.CD80_high
    candidates=cells[needed&cells.inside_final_fiber].copy()
    fibers=pd.read_parquet(INTEGRATION/'final_fiber_manifest.parquet')
    assert len(fibers)==27904 and fibers.canonical_fiber_id.is_unique
    flookup=fibers.set_index('canonical_fiber_id')
    normal=pd.read_parquet(NORMAL).set_index('canonical_fiber_id')
    pieces=pd.read_csv(DMDROOT/'tables/piece_manifest.csv').set_index('piece_id')
    tiles={s:pd.read_csv(Path('/Volumes/WALLE/ucla_dmd_cell_features_20260812/per_slide')/s/'tile_manifest.csv').set_index('tile_id') for s in cells.slide_id.unique()}
    rows=[]; ledger=[]
    for piece,g in candidates.groupby('piece_id',observed=True,sort=True):
        arrays={}; isdmd=g.clinical_status.iloc[0]=='DMD'
        if isdmd:
            p=pieces.loc[piece]; path=DMDROOT/p.label_path
            labels=tifffile.imread(path)
            cy,cx=int(p.piece_crop_y0_slide_level0_px),int(p.piece_crop_x0_slide_level0_px)
            # Record the exact native label file used for pixel-overlap checks.
            ledger.append({'piece_id':piece,'path':str(path),'sha256':sha(path),'cells_checked':len(g)})
        cache={}
        for row in g.itertuples():
            yy,xx=cell_pixels(row,tiles,arrays)
            f=flookup.loc[row.canonical_fiber_id]
            if isdmd:
                y,x=yy-cy,xx-cx
                valid=(y>=0)&(x>=0)&(y<labels.shape[0])&(x<labels.shape[1])
                inside=np.zeros(len(y),bool)
                inside[valid]=labels[y[valid],x[valid]]==int(f.piece_label_id)
                sy,sx=int(np.floor(row.centroid_y_px-cy)),int(np.floor(row.centroid_x_px-cx))
                assert int(labels[sy,sx])==int(f.piece_label_id),row.cell_id
            else:
                n=normal.loc[row.canonical_fiber_id]
                path=Path(n.source_mask_path)
                if str(path) not in cache:cache[str(path)]=tifffile.imread(path)
                mask=cache[str(path)]
                # Strict masks are piece-local; exact accepted masks are bbox-local.
                if str(n.source_kind)=='review_queue_exact_mask':
                    oy=float(n.centroid_y_slide_px)-float(n.centroid_y_piece_px)+float(n.mask_y0_piece_px)
                    ox=float(n.centroid_x_slide_px)-float(n.centroid_x_piece_px)+float(n.mask_x0_piece_px)
                else:
                    oy=float(n.centroid_y_slide_px)-float(n.centroid_y_piece_px)
                    ox=float(n.centroid_x_slide_px)-float(n.centroid_x_piece_px)
                y,x=yy-int(round(oy)),xx-int(round(ox))
                valid=(y>=0)&(x>=0)&(y<mask.shape[0])&(x<mask.shape[1])
                inside=np.zeros(len(y),bool)
                inside[valid]=mask[y[valid],x[valid]]==int(n.source_mask_label_value)
                sy=int(np.floor(row.centroid_y_px-oy));sx=int(np.floor(row.centroid_x_px-ox))
                assert 0<=sy<mask.shape[0] and 0<=sx<mask.shape[1],row.cell_id
                assert int(mask[sy,sx])==int(n.source_mask_label_value),row.cell_id
            fraction=float(inside.mean())
            rows.append({'cell_id':row.cell_id,'cell_mask_pixels':len(inside),
                         'cell_mask_pixels_in_assigned_fiber':int(inside.sum()),
                         'cell_mask_fraction_in_assigned_fiber':fraction,
                         'overlap_status':'measured_native_masks'})
        print(f'  {piece}: {len(g)} candidate cell masks checked.',flush=True)
    overlap=pd.DataFrame(rows)
    assert overlap.cell_id.is_unique and len(overlap)==len(candidates)
    cells=cells.merge(overlap,on='cell_id',how='left',validate='one_to_one')
    cells['interior_centroid_2um']=cells.inside_final_fiber&cells.signed_boundary_distance_um.le(-2)
    cells['strict_interior_2um_and_cell80']=cells.interior_centroid_2um&cells.cell_mask_fraction_in_assigned_fiber.ge(.8)
    cells['containment_interpretation']='Provisional spatial assignment in a 2D section; not visually confirmed intrafiber residence'
    cells.to_csv(OUT / 'Intra_myofiber/Cell_gate_and_fiber_assignments.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    cells.to_parquet(OUT / 'reproducibility/cell_gate_and_geometry_inputs.parquet',index=False)
    pd.DataFrame(ledger).to_csv(OUT/'validation/DMD_native_fiber_mask_read_ledger.csv',index=False)
    summarize(cells,fibers)
    jsave(OUT/'validation/containment_checks.json',{'status':'passed','candidate_cells_checked':len(candidates),
          'native_cell_mask_pixel_counts_match_saved_areas':True,'DMD_centroid_label_matches_saved_assignment':True,
          'cell_overlap_fraction_within_0_and_1':bool(cells.cell_mask_fraction_in_assigned_fiber.dropna().between(0,1).all()),
          'split_fiber_merge_review':'pending; no unreviewed merging performed',
          'whole_fiber_unit':'one row per existing canonical_fiber_id; original masks untouched',
          'spatial_results_status':'provisional; segmentation and positive/negative marker calibration not independently validated'})
    print('All tables and native-mask containment checks complete.',flush=True)


def populations(cells):
    result={}
    k,b=cells.GZMK_positive,cells.GZMB_positive
    parents={'CD45_CD4_high':cells.CD45_CD4_high,'CD45_CD8_high':cells.CD45_CD8_high,
             'CD4_T_like_with_CD45_CD4_high':cells.CD45_CD4_high&cells.lineage_cell_type.astype(str).eq('CD4 T-like'),
             'CD8_T_like_with_CD45_CD8_high':cells.CD45_CD8_high&cells.lineage_cell_type.astype(str).eq('CD8 T-like')}
    for parent,p in parents.items():
        for state,flag in [('all',np.ones(len(cells),bool)),('GZMK_positive_including_dual',k),
                           ('GZMB_positive_including_dual',b),('K_positive_B_negative',k&~b),
                           ('K_negative_B_positive',~k&b),('K_positive_B_positive',k&b)]:
            result[parent+'__'+state]=p&flag
    result['CD80_high_all_cells']=cells.CD80_high
    result['CD45_high_CD80_high']=cells.CD45_high&cells.CD80_high
    result['original_CD4_T_like']=cells.lineage_cell_type.astype(str).eq('CD4 T-like')
    result['original_CD8_T_like']=cells.lineage_cell_type.astype(str).eq('CD8 T-like')
    return result


def summarize(cells,fibers):
    groups=populations(cells)
    places={'all_retained_cells':np.ones(len(cells),bool),'centroid_inside':cells.inside_final_fiber,
            'centroid_at_least_2um_inside':cells.interior_centroid_2um,
            'centroid_2um_inside_and_cell_mask_80percent_inside':cells.strict_interior_2um_and_cell80}
    donor_meta=cells[['donor_id','clinical_status','group','severity']].drop_duplicates()
    f=fibers.copy();f['donor_id']=f.specimen_id.astype(str)
    den=f.groupby('donor_id',observed=True).agg(analyzed_fiber_N=('canonical_fiber_id','size'),analyzed_fiber_area_um2=('area_um2','sum'))
    rows=[]
    fiber_counts=f[['canonical_fiber_id','donor_id','clinical_status','piece_id','piece_label_id','area_um2',
                    'large_profile_model_derived','exclude_all_model_large_profile_sensitivity']].copy()
    for group,flag in groups.items():
        all_counts=cells.loc[flag].groupby('donor_id',observed=True).size()
        for place,where in places.items():
            subset=cells.loc[flag&where]
            n=subset.groupby('donor_id',observed=True).size()
            nf=subset.groupby('donor_id',observed=True).canonical_fiber_id.nunique() if place!='all_retained_cells' else None
            for d in donor_meta.itertuples():
                count=int(n.get(d.donor_id,0));total=int(all_counts.get(d.donor_id,0))
                rows.append({'donor_id':d.donor_id,'group':d.group,'severity':d.severity,'population':group,
                             'location_rule':place,'cell_N':count,'all_retained_population_N':total,
                             'percent_of_population':100*count/total if total else np.nan,
                             'fibers_with_population_N':int(nf.get(d.donor_id,0)) if nf is not None else np.nan,
                             'analyzed_fiber_N':int(den.loc[d.donor_id,'analyzed_fiber_N']),
                             'analyzed_fiber_area_mm2':float(den.loc[d.donor_id,'analyzed_fiber_area_um2']/1e6),
                             'status':'provisional_mask_based_count'})
            if place in ['centroid_inside','centroid_2um_inside_and_cell_mask_80percent_inside']:
                byfiber=subset.groupby('canonical_fiber_id',observed=True).size()
                stem='strict' if place.startswith('centroid_2um') else 'centroid'
                fiber_counts[stem+'__'+group]=fiber_counts.canonical_fiber_id.map(byfiber).fillna(0).astype(int)
                assert int(fiber_counts[stem+'__'+group].sum())==len(subset)
    donor=pd.DataFrame(rows)
    donor.to_csv(OUT/'Intra_myofiber/Counts_by_donor_population_and_rule.csv',index=False)
    pooled=donor.groupby(['group','population','location_rule'],observed=True).agg(
        cell_N=('cell_N','sum'),all_retained_population_N=('all_retained_population_N','sum'),
        donors_with_cells=('cell_N',lambda z:int((z>0).sum())),donor_N=('donor_id','nunique')).reset_index()
    pooled['percent_of_population']=100*pooled.cell_N/pooled.all_retained_population_N.replace(0,np.nan)
    pooled['status']='provisional_mask_based_count'
    pooled.to_csv(OUT/'Intra_myofiber/Pooled_counts_by_group_population_and_rule.csv',index=False)
    fiber_counts.to_csv(OUT/'Intra_myofiber/Counts_per_existing_whole_fiber.csv',index=False)
    requested=cells.CD45_CD4_high|cells.CD45_CD8_high|cells.CD80_high
    review=cells.loc[requested&cells.inside_final_fiber].copy()
    review['manual_cell_identity_review']='pending'
    review['manual_outer_fiber_boundary_review']='pending'
    review.to_csv(OUT/'Intra_myofiber/Candidate_cells_for_image_review.csv',index=False)
    poscols=[k for k in fiber_counts if k.startswith('centroid__')]
    queue=fiber_counts[fiber_counts[poscols].sum(axis=1)>0].copy()
    queue['whole_fiber_group_id']=queue.canonical_fiber_id
    queue['outer_boundary_review']='pending'
    queue['merge_with_canonical_fiber_ids']=''
    queue['review_note']='No automatic merge based on immune-marker staining; verify the external anatomical boundary'
    queue.to_csv(OUT/'Intra_myofiber/Whole_fiber_boundary_review_queue.csv',index=False)
    queue[queue.donor_id.eq('8011')].to_csv(OUT/'Intra_myofiber/Sample_8011_whole_fiber_review_queue.csv',index=False)
    print(pooled[pooled.location_rule.isin(['centroid_inside','centroid_2um_inside_and_cell_mask_80percent_inside'])].to_string(index=False),flush=True)


def refresh_summaries():
    cells=pd.read_parquet(OUT/'reproducibility/cell_gate_and_geometry_inputs.parquet')
    cells['GZMK_q99_high']=cells.GZMK_positive
    cells['GZMB_q99_high']=cells.GZMB_positive
    cells['GZMK_absolute_positivity_calibration_status']=np.where(cells.slide_id.astype(str).isin(['F2','F5']),
            'abstained_in_original_calibration; q99-high sensitivity call only',
            'frozen_q99-high operational call; not a new absolute calibration')
    cells.to_parquet(OUT/'reproducibility/cell_gate_and_geometry_inputs.parquet',index=False)
    cells.to_csv(OUT/'Intra_myofiber/Cell_gate_and_fiber_assignments.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    fibers=pd.read_parquet(INTEGRATION/'final_fiber_manifest.parquet')
    summarize(cells,fibers)
    # A shared legacy parent is a review clue, not proof that current profiles belong to one fiber.
    d= fibers[fibers.clinical_status.eq('DMD')&fibers.dominant_legacy_instance_uid.notna()].copy()
    g=d.groupby(['piece_id','dominant_legacy_instance_uid'],observed=True)
    d['current_fibers_sharing_legacy_parent_N']=g.canonical_fiber_id.transform('size')
    linked=d[d.current_fibers_sharing_legacy_parent_N>1].copy()
    linked['interpretation']='Shared old-segmentation parent only; possible split or old undersegmentation. Not an approved merge.'
    linked['manual_outer_boundary_review']='pending'
    linked.to_csv(OUT/'Intra_myofiber/Global_possible_split_profiles_review.csv',index=False)
    queue=pd.read_csv(OUT/'Intra_myofiber/Whole_fiber_boundary_review_queue.csv',dtype={'donor_id':str})
    queue=queue.merge(d[['canonical_fiber_id','dominant_legacy_instance_uid','current_fibers_sharing_legacy_parent_N']],
                      on='canonical_fiber_id',how='left',validate='one_to_one')
    queue['possible_split_or_old_undersegmentation']=queue.current_fibers_sharing_legacy_parent_N.gt(1)
    queue.to_csv(OUT/'Intra_myofiber/Whole_fiber_boundary_review_queue.csv',index=False)
    queue[queue.donor_id.eq('8011')].to_csv(OUT/'Intra_myofiber/Sample_8011_whole_fiber_review_queue.csv',index=False)
    full=pd.read_parquet(OUT/'UMAP/All_retained_cells_marker_expression.parquet')
    donor_severity=cells[['donor_id','severity']].drop_duplicates().set_index('donor_id').severity
    full['severity']=full.donor_id.astype(str).map(donor_severity)
    assert full.severity.notna().all()
    full.to_parquet(OUT/'UMAP/All_retained_cells_marker_expression.parquet',index=False)
    full.to_csv(OUT/'UMAP/All_retained_cells_marker_expression.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    jsave(OUT/'validation/global_scope_checks.json',{'donors':int(cells.donor_id.nunique()),'retained_cells':len(cells),
          'whole_fiber_rows':len(fibers),'DMD_fiber_rows':int(fibers.clinical_status.eq('DMD').sum()),
          'Healthy_fiber_rows':int(fibers.clinical_status.eq('Normal').sum()),
          'all_retained_cells_in_spatial_counts':True,'each_cell_counted_once_within_each_population':True,
          'direct_CD45_CD4_and_CD45_CD8_parent_sets_may_overlap':True,
          'T_like_only_sensitivities_exported':True,'granzyme_q99_sensitivity_calls_not_absolute_calibration':True,
          'shared_legacy_parent_review_groups':int(linked[['piece_id','dominant_legacy_instance_uid']].drop_duplicates().shape[0]),
          'fibers_in_shared_legacy_parent_review_groups':len(linked),'automatic_unreviewed_merges':False})


def verify_exports():
    """Read saved files back, and independently accumulate selected counts cell by cell."""
    u=pd.read_parquet(UMAP)
    exported=pd.read_csv(OUT/'UMAP/UMAP_cells_marker_expression_coordinates.csv',dtype={'cell_id':str,'donor_id':str},float_precision='round_trip')
    assert len(exported)==75664 and exported.cell_id.is_unique
    assert np.array_equal(exported.cell_id,u.cell_id)
    np.testing.assert_allclose(exported[['UMAP1','UMAP2']],u[['UMAP1','UMAP2']],rtol=1e-7,atol=1e-7)
    original=pd.read_parquet(ID/'cell_marker_source_signed_means.parquet').set_index('cell_id')
    for marker,column in [('CD45_555','marker_CD45_555'),('CD4_750','marker_CD4_750'),('CD8a_555','marker_CD8a_555'),('CD80','marker_CD80')]:
        np.testing.assert_allclose(exported[column],original.loc[exported.cell_id,marker],rtol=1e-7,atol=1e-7)
    allcells=pd.read_csv(OUT/'UMAP/All_retained_cells_marker_expression.csv.gz',usecols=['cell_id','donor_id','included_in_UMAP','severity','UMAP1','UMAP2'],dtype={'donor_id':str})
    assert len(allcells)==205675 and allcells.cell_id.is_unique and allcells.severity.notna().all()
    assert int(allcells.included_in_UMAP.sum())==75664
    assert allcells.loc[~allcells.included_in_UMAP,['UMAP1','UMAP2']].isna().all().all()
    cells=pd.read_parquet(OUT/'reproducibility/cell_gate_and_geometry_inputs.parquet')
    pooled=pd.read_csv(OUT/'Intra_myofiber/Pooled_counts_by_group_population_and_rule.csv')
    groups=populations(cells)
    for name in ['CD45_CD4_high__all','CD45_CD8_high__all','CD80_high_all_cells',
                 'CD8_T_like_with_CD45_CD8_high__K_positive_B_positive']:
        flags=groups[name].to_numpy();strict=cells.strict_interior_2um_and_cell80.to_numpy()
        for group in ['DMD','Healthy']:
            actual=sum(bool(flag and inside and g==group) for flag,inside,g in zip(flags,strict,cells.group))
            reported=pooled[(pooled.population==name)&(pooled.group==group)&(pooled.location_rule=='centroid_2um_inside_and_cell_mask_80percent_inside')].cell_N.item()
            assert actual==reported,(name,group,actual,reported)
    jsave(OUT/'validation/saved_export_verification.json',{'status':'passed','saved_CSV_coordinates_match_original':True,
          'saved_CSV_key_marker_values_match_source':True,'saved_full_cell_matrix_IDs_and_UMAP_missingness_match':True,
          'independent_cellwise_count_checks':8,'no_numeric_rounding_applied_to_source_values':True})
    selected=pooled[pooled.location_rule.isin(['centroid_inside','centroid_2um_inside_and_cell_mask_80percent_inside'])]
    quick=selected.pivot(index='population',columns=['group','location_rule'],values='cell_N')
    quick.columns=[group+'__'+rule for group,rule in quick.columns]
    quick=quick.reset_index()
    quick['interpretation']='Provisional counts within segmented 2D profiles. Marker-high is not newly calibrated absolute positivity.'
    quick.to_csv(OUT/'Intra_myofiber/Intrafiber_quick_summary.csv',index=False)


def examples_8011():
    """Show the two masks containing multiple requested T-cell candidates, not an inferred screenshot match."""
    import zarr
    cells=pd.read_parquet(OUT/'reproducibility/cell_gate_and_geometry_inputs.parquet')
    candidates=cells[cells.donor_id.eq('8011')&cells.CD45_CD8_high&cells.inside_final_fiber]
    fids=candidates.groupby('canonical_fiber_id',observed=True).size()
    fids=fids[fids>=2].index.tolist()
    assert len(fids)==2
    a=ad.read_h5ad(H5,backed='r');var=a.var.copy();obs=a.obs.copy();obs.index=obs.cell_id.astype(str);a.file.close()
    inst=pd.read_csv(DMDROOT/'tables/dmd_review_informed_instances.csv').set_index('canonical_fiber_id')
    pieces=pd.read_csv(DMDROOT/'tables/piece_manifest.csv').set_index('piece_id')
    guard=pd.read_csv(ROOT/'exports/collaborator_followup_20260922/v1/tissue/tables/source_ome_guard.csv').set_index('slide_id')
    markers=['DAPI','CD44_488','Collagen Type IV','CD45_555','CD8a_555','CD4_750','Granzyme K_647','Granzyme B','CD80']
    short=['DAPI (raw)','CD44','Collagen IV','CD45','CD8a','CD4','GZMK','GZMB','CD80']
    ledger=[];bounds=[];selected=[]
    for num,fid in enumerate(fids,1):
        f=inst.loc[fid];piece=f.piece_id;p=pieces.loc[piece];g=guard.loc[p.slide_id]
        targets=candidates[candidates.canonical_fiber_id.eq(fid)].copy().sort_values('cell_id')
        x0f=int(p.piece_crop_x0_slide_level0_px)+int(f.bbox_x0_piece_px)
        y0f=int(p.piece_crop_y0_slide_level0_px)+int(f.bbox_y0_piece_px)
        x1f=int(p.piece_crop_x0_slide_level0_px)+int(f.bbox_x1_piece_px)
        y1f=int(p.piece_crop_y0_slide_level0_px)+int(f.bbox_y1_piece_px)
        width=max(round(200/PIX),max(x1f-x0f,y1f-y0f)+2*round(20/PIX))
        x0=int(np.clip(round((x0f+x1f-width)/2),0,int(g.size_x)-width))
        y0=int(np.clip(round((y0f+y1f-width)/2),0,int(g.size_y)-width))
        assert x0<=x0f<x1f<=x0+width and y0<=y0f<y1f<=y0+width
        labels=tifffile.imread(DMDROOT/p.label_path)
        labelcrop=np.zeros((width,width),dtype=labels.dtype)
        cx,cy=int(p.piece_crop_x0_slide_level0_px),int(p.piece_crop_y0_slide_level0_px)
        xa,xb=max(x0,cx),min(x0+width,cx+labels.shape[1]);ya,yb=max(y0,cy),min(y0+width,cy+labels.shape[0])
        labelcrop[ya-y0:yb-y0,xa-x0:xb-x0]=labels[ya-cy:yb-cy,xa-cx:xb-cx]
        targetfiber=labelcrop==int(f.piece_label_id)
        assert int(targetfiber.sum())==int(f.area_px)
        source=Path(g.ome_source_path);stat=source.stat()
        assert stat.st_size==int(g.size_bytes) and stat.st_mtime_ns==int(g.mtime_ns)
        store=tifffile.imread(source,aszarr=True);root=zarr.open(store,mode='r');arr=root['0'] if isinstance(root,zarr.Group) else root
        signals={};display={}
        try:
            for marker in markers:
                if marker=='DAPI':
                    si=int(obs.loc[targets.iloc[0].cell_id,'dapi_anchor_channel_index']);bi=None;factor=None
                    signal=np.asarray(arr[si,y0:y0+width,x0:x0+width],dtype=np.float32)
                else:
                    v=var.loc[marker];assert bool(v.pair_valid_flag)
                    si,bi,factor=int(v.signal_channel_index),int(v.background_channel_index),float(v.exposure_scale)
                    signal=np.asarray(arr[si,y0:y0+width,x0:x0+width],dtype=np.float32)-factor*np.asarray(arr[bi,y0:y0+width,x0:x0+width],dtype=np.float32)
                signals[marker]=signal
                hi=max(float(np.quantile(signal,.998)),1)
                display[marker]=np.clip(signal/hi,0,1)
                bounds.append({'example':num,'marker':marker,'display_low':0,'display_high':hi,
                               'display_rule':'Within-example 99.8th percentile; display only, not quantitative cross-example scaling'})
                ledger.append({'example':num,'canonical_fiber_id':fid,'donor_id':'8011','piece_id':piece,
                               'x0_slide_px':x0,'y0_slide_px':y0,'width_px':width,'actual_width_um':width*PIX,
                               'marker':marker,'signal_channel_index':si,'background_channel_index':bi,
                               'exposure_scale':factor,'source_ome':str(source),'source_size_bytes':stat.st_size,
                               'source_mtime_ns':stat.st_mtime_ns,'signed_crop_sha256':hashlib.sha256(signal.tobytes()).hexdigest()})
        finally:store.close()
        fig,axes=plt.subplots(2,5,figsize=(15,6.5));extent=[0,width*PIX,width*PIX,0]
        for ax,marker,label in zip(axes.ravel(),markers,short):
            ax.imshow(display[marker],cmap='gray',vmin=0,vmax=1,extent=extent)
            ax.text(.025,.975,label,transform=ax.transAxes,color='white',va='top',fontsize=9,
                    bbox={'facecolor':'black','alpha':.65,'edgecolor':'none','pad':2})
        rgb=np.zeros((width,width,3),float)
        rgb[:,:,2]=display['DAPI']*.65+display['CD45_555']*.3
        rgb[:,:,1]=display['CD44_488']*.55+display['Granzyme K_647']*.7
        rgb[:,:,0]=display['Granzyme B']*.75+display['CD8a_555']*.65
        axes.ravel()[-1].imshow(np.clip(rgb,0,1),extent=extent)
        axes.ravel()[-1].text(.025,.975,'Merge',transform=axes.ravel()[-1].transAxes,color='white',va='top',fontsize=9,
                            bbox={'facecolor':'black','alpha':.65,'edgecolor':'none','pad':2})
        xy=(np.arange(width)+.5)*PIX
        for ax in axes.ravel():
            ax.contour(xy,xy,targetfiber,levels=[.5],colors=['white'],linewidths=.65)
            for i,t in enumerate(targets.itertuples()):
                x,y=t.centroid_x_um-x0*PIX,t.centroid_y_um-y0*PIX
                ax.plot(x,y,marker='o',markersize=9,fillstyle='none',markeredgewidth=.8,color='#55E5FF')
                ax.annotate(chr(65+i),xy=(x,y),xytext=(x+6,y-8 if i%2==0 else y+9),
                            color='#55E5FF',fontsize=8,arrowprops={'arrowstyle':'-','color':'#55E5FF','lw':.6})
            ax.plot([10,60],[width*PIX-12]*2,c='white',lw=2)
            ax.text(35,width*PIX-15,'50 µm',color='white',ha='center',fontsize=7)
            ax.set_xlim(0,width*PIX);ax.set_ylim(width*PIX,0);ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'8011 · {piece} · fiber {int(f.piece_label_id)} · boundary review',fontsize=12)
        fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.92,wspace=.035,hspace=.035)
        savefig(fig,f'8011_native_boundary_review_{num}')
        for i,t in enumerate(targets.itertuples()):
            record=t._asdict();record.update(example=num,image_label=chr(65+i),
                  screenshot_match_confirmed=False,outer_whole_fiber_boundary_review='pending')
            selected.append(record)
        print(f'Example {num}: {fid}; entire {int(f.area_px):,}-pixel mask shown.',flush=True)
    pd.DataFrame(ledger).to_csv(OUT/'validation/8011_native_image_read_ledger.csv',index=False)
    pd.DataFrame(bounds).to_csv(OUT/'validation/8011_image_display_bounds.csv',index=False)
    pd.DataFrame(selected).to_csv(OUT/'Intra_myofiber/Sample_8011_image_key.csv',index=False)


def finalize():
    import importlib.metadata as metadata
    import platform
    assert json.loads((OUT/'validation/saved_export_verification.json').read_text())['status']=='passed'
    assert json.loads((OUT/'validation/containment_checks.json').read_text())['status']=='passed'
    checks=json.loads((OUT/'validation/export_checks.json').read_text())
    checks['UMAP_count_legend_visual_review']='passed'
    checks['8011_native_review_image_layouts_visually_checked']=True
    checks['manual_outer_fiber_boundary_adjudication']='not completed; counts remain provisional'
    jsave(OUT/'validation/export_checks.json',checks)
    jsave(OUT/'reproducibility/software_versions.json',{'python':platform.python_version(),
           **{p:metadata.version(p) for p in ['anndata','numpy','pandas','pyarrow','tifffile','zarr','matplotlib']}})
    shutil.copy2(Path(__file__),OUT/'reproducibility'/Path(__file__).name)
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.sha256')
    (OUT/'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files))


def mirror():
    results={}
    for name,root in {'OneDrive':Path('/Users/lownikj/Library/CloudStorage/OneDrive-Cedars-SinaiHealthSystem/codex_output/ucla_dmd'),
                      'Shared_drive':Path('/Volumes/home/codex_output/ucla_dmd')}.items():
        try:
            if name=='Shared_drive' and not os.path.ismount('/Volumes/home'):raise RuntimeError('/Volumes/home is not mounted')
            if name=='OneDrive' and not root.parents[1].is_dir():raise RuntimeError('OneDrive provider unavailable')
            if not root.is_dir():raise RuntimeError('Existing output destination unavailable')
            dest=root/REL;files=sorted(p for p in OUT.rglob('*') if p.is_file())
            for p in files:
                t=dest/p.relative_to(OUT)
                if t.exists() and sha(t)!=sha(p):raise RuntimeError(f'Divergent destination file preserved: {t}')
            for p in files:
                t=dest/p.relative_to(OUT);t.parent.mkdir(parents=True,exist_ok=True)
                if not t.exists():shutil.copy2(p,t)
                assert sha(t)==sha(p)
            results[name]={'folder':str(dest),'files_verified':len(files),'method':'SHA-256','cloud_sync_confirmed':False}
        except Exception as e:results[name]={'pending':str(e)}
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finalize-reviewed',action='store_true');parser.add_argument('--mirror',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--examples-8011',action='store_true');parser.add_argument('--refresh-summaries',action='store_true');parser.add_argument('--verify-exports',action='store_true')
    args=parser.parse_args()
    if args.mirror:mirror()
    elif args.finalize_reviewed:finalize()
    elif args.examples_8011:examples_8011()
    elif args.refresh_summaries:refresh_summaries()
    elif args.verify_exports:verify_exports()
    else:tables(resume=args.resume)
