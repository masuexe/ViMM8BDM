"""Replay image-based aim proposals without game input or scene feedback."""
import io,json,struct,time,collections,statistics,argparse
from pathlib import Path
from PIL import Image
from vimm8bdm.aim_tracking import StableAim,filter_candidates
from vimm8bdm.visual_labels import source_identity
from vimm8bdm.recording import file_digest
from vimm8bdm.visual_metrics import box_iou
parser=argparse.ArgumentParser(description='Replay mouse-only proposals from a checked detector scan without game input')
parser.add_argument('--recording',type=Path,required=True)
parser.add_argument('--scan',type=Path,required=True)
parser.add_argument('--reviewed-images',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--inspect-frames',type=int,nargs='*',default=[])
args=parser.parse_args()
root=args.output
if any(source.resolve()==root.resolve() or source.resolve() in root.resolve().parents for source in [args.recording,args.scan,args.reviewed_images]):
 raise ValueError('Create replay results outside the source directories')
root.mkdir(exist_ok=False,parents=True)
raw=args.recording
source,hashes=source_identity(raw)
scan=args.scan
prior=json.loads((scan/'protocol.local.json').read_text())
if (source,hashes)!=(prior['source_session_id'],prior['source_hashes']):raise ValueError('The recording differs from its detector scan')
predictions=[json.loads(s) for s in (scan/'frames.local.jsonl').read_text().splitlines()]
index=[json.loads(s) for s in (raw/'frames.jsonl').read_text().splitlines()]
if len(index)!=len(predictions) or any(a['frame']!=b['frame'] for a,b in zip(index,predictions)):raise ValueError('The scan does not cover all recording frames in order')
protocol=dict(format='vimm8bdm-offline-aim-probe-v1',source_session_id=source,source_hashes=hashes,checkpoint_sha256=prior['checkpoint_sha256'],aim_code_sha256=file_digest(Path('src/vimm8bdm/aim_tracking.py')),score_threshold=.8,magenta_fraction=.03,magenta_vertical_extent=.45,confirmation_frames=3,minimum_aspect_ratio=.2,maximum_aspect_ratio=1.1,source_frame_step=2,independent_test=False,closed_loop_simulation=False,input_sent=False)
(root/'protocol.local.json').write_text(json.dumps(protocol,indent=2))
tracker=StableAim();cases=[];filter_times=[];known={}
with (raw/'frames.vrec').open('rb') as archive:
 for p,row in zip(predictions,index):
  if row['frame']%2 and row['frame'] not in args.inspect_frames:continue
  archive.seek(row['offset']);n=struct.unpack('<I',archive.read(4))[0];image=Image.open(io.BytesIO(archive.read(n))).convert('RGB')
  started=time.perf_counter();accepted,rejected=filter_candidates(image,p['predictions']);filter_times.append((time.perf_counter()-started)*1000)
  if row['frame'] in args.inspect_frames:known[str(row['frame'])]=dict(eligible=accepted,rejected=rejected)
  if row['frame']%2:continue
  action,detail=tracker.step(accepted,row['width'],row['height'],row['capture_started_ns'])
  if action.keys or action.attack:raise ValueError('Aim replay must not propose keys or attack')
  cases.append(dict(frame=row['frame'],eligible=accepted,rejected=rejected,aim=detail,mouse_counts=[action.mouse_dx,action.mouse_dy]))
(root/'frames.local.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases))
validation=args.reviewed_images;
metadata=json.loads((validation/'dataset.json').read_text())
if metadata['source_session_id']!=source:raise ValueError('The reviewed images belong to a different recording')
coco=json.loads((validation/'annotations.coco.json').read_text());truth={a['image_id']:a['bbox'] for a in coco['annotations']};checks=[]
for r in coco['images']:
 if file_digest(validation/r['file_name'])!=metadata['image_sha256'][str(r['id'])]:raise ValueError('A reviewed image changed after export')
 with Image.open(validation/r['file_name']) as im:good,bad=filter_candidates(im.convert('RGB'),predictions[r['id']]['predictions'])
 checks.append(dict(frame=r['id'],positive=r['id'] in truth,eligible=good,located=any(box_iou(p['box'],truth[r['id']])>=.5 for p in good) if r['id'] in truth else False))
if source_identity(raw)!=(source,hashes):raise ValueError('The recording changed during replay')
summary=dict(frames=len(cases),mouse_proposals=sum(any(c['mouse_counts']) for c in cases),eligible_frames=sum(bool(c['eligible']) for c in cases),reasons=dict(collections.Counter(c['aim']['reason'] for c in cases)),positive_validation_locations=sum(c['located'] for c in checks),positive_validation_images=len(truth),negative_validation_candidates=sum(not c['positive'] and bool(c['eligible']) for c in checks),maximum_mouse_counts=[max(abs(c['mouse_counts'][axis]) for c in cases) for axis in [0,1]],appearance_filter_median_ms=statistics.median(filter_times),appearance_filter_maximum_ms=max(filter_times))
report=dict(protocol=protocol,summary=summary,known_failure_cases=known,validation_cases=checks,input_sent=False,source_unchanged=True,combat_accuracy_proved=False)
(root/'report.local.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2));print(json.dumps(known,indent=2))
