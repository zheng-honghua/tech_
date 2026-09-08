"""Export ambiguous frame instances for explicit human label confirmation."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    target = Path(args.output_dir)
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    records = []
    for number in range(6, 11):
        folder = Path(args.results_root) / 'multi-02' / f'frame-{number:02d}'
        results = json.loads((folder / 'results-v2.json').read_text(encoding='utf-8'))
        for result in results:
            filename = f"{result['object_id']}.png"
            shutil.copyfile(folder / result['crop_image'], target / filename)
            records.append({'frame_id': result['frame_id'], 'object_id': result['object_id'],
                            'bbox_px': result['bbox_px'], 'image': filename,
                            'label_id': None, 'reviewed': False})
    (target / 'labels.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'{len(records)} instances exported to {target}')


if __name__ == '__main__':
    main()
