"""Check a private recording and export an image-and-event preview."""
import argparse
import base64
import collections
import io
import json
from pathlib import Path
import statistics
import struct

from .recording import MAGIC


def review(directory, preview=None):
    from PIL import Image
    manifest = json.loads((directory / 'session.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    frames = [json.loads(line) for line in (directory / 'frames.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    if not summary['complete'] or summary['dropped_items']:
        raise ValueError('The recording is incomplete')
    if len(frames) != summary['frames'] or len(events) != summary['events']:
        raise ValueError('The summary does not match the saved record count')
    if not events or events[-1]['type'] != 'recording_end':
        raise ValueError('The recording has no final event')
    if [e['sequence'] for e in events] != list(range(1, len(events) + 1)):
        raise ValueError('The event sequence has a gap')
    by_segment = collections.defaultdict(list)
    for event in events:
        by_segment[event['segment']].append(event)
    items = []
    durations = []
    gaps = []
    last = None
    stride = max(1, (len(frames) + 119) // 120)
    with (directory / 'frames.vrec').open('rb') as archive:
        if archive.read(len(MAGIC)) != MAGIC:
            raise ValueError('The image archive signature is invalid')
        for i, frame in enumerate(frames):
            if frame['frame'] != i or frame['offset'] != archive.tell():
                raise ValueError('The image index is discontinuous')
            header = archive.read(4)
            if len(header) != 4 or struct.unpack('<I', header)[0] != frame['png_bytes']:
                raise ValueError('The image length does not match its index')
            raw = archive.read(frame['png_bytes'])
            if len(raw) != frame['png_bytes']:
                raise ValueError('The image archive is truncated')
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (frame['width'], frame['height']):
                    raise ValueError('The image dimensions do not match the index')
                image.verify()
            if frame['capture_finished_ns'] < frame['capture_started_ns']:
                raise ValueError('The capture interval is invalid')
            durations.append((frame['capture_finished_ns'] - frame['capture_started_ns']) / 1e6)
            if last and frame['segment'] == last['segment']:
                if frame['capture_started_ns'] <= last['capture_started_ns']:
                    raise ValueError('The image clock did not advance')
                gaps.append((frame['capture_started_ns'] - last['capture_started_ns']) / 1e6)
            last = frame
            if preview and i % stride == 0:
                end = frames[i + 1]['capture_finished_ns'] if i + 1 < len(frames) and frames[i + 1]['segment'] == frame['segment'] else frame['capture_finished_ns']
                following = [e for e in by_segment[frame['segment']]
                             if frame['capture_finished_ns'] <= e['time_ns'] < end]
                items.append({'image': base64.b64encode(raw).decode('ascii'),
                              'frame': i, 'segment': frame['segment'],
                              'capture_ms': durations[-1], 'following_events': following})
        if archive.read(1):
            raise ValueError('The archive contains unindexed images')
    report = {'frames': len(frames), 'events': len(events), 'segments': summary['segments'],
              'event_types': dict(collections.Counter(e['type'] for e in events)),
              'capture_ms_median': statistics.median(durations) if durations else None,
              'capture_ms_max': max(durations) if durations else None,
              'same_segment_gap_ms_max': max(gaps) if gaps else None,
              'archive_integrity': 'pass', 'input_sent': manifest['input_sent'],
              'training_ready': manifest['training_ready'],
              'label_alignment': 'Receipt-clock data. Visual alignment requires operator review.'}
    if preview:
        # Escape markup delimiters before embedding local JSON in a script element.
        data = json.dumps(items).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        html = '''<!doctype html><meta charset="utf-8"><title>Recording review</title>
<style>body{font:16px sans-serif;background:#171717;color:#eee;max-width:1000px;margin:20px auto}img{max-width:100%;image-rendering:pixelated}input{width:100%}pre{white-space:pre-wrap}</style>
<h1>Recording review</h1><p>Move the slider to inspect sampled images. Events follow capture completion until the next frame in the same segment. This preview does not send input.</p>
<input id="position" type="range" min="0" value="0"><img id="picture" alt="Recorded game frame"><pre id="details"></pre>
<script id="data" type="application/json">DATA</script><script>
const rows=JSON.parse(document.getElementById('data').textContent);
const slider=document.getElementById('position');slider.max=Math.max(0,rows.length-1);
function draw(){const row=rows[Number(slider.value)];if(!row)return;document.getElementById('picture').src='data:image/png;base64,'+row.image;const {image,...metadata}=row;document.getElementById('details').textContent=JSON.stringify(metadata,null,2)}
slider.addEventListener('input',draw);draw();</script>'''.replace('DATA', data)
        with preview.open('x', encoding='utf-8') as output:
            output.write(html)
    return report


def main():
    parser = argparse.ArgumentParser(description='Check a private human recording')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--preview', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.run_dir, args.preview), indent=2))


if __name__ == '__main__':
    main()
