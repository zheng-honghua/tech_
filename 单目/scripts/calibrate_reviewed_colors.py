"""Build a local D415 colour profile from explicitly reviewed blue/cyan crops."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--base-config', default='config/default.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    samples = {'blue': [], 'cyan': []}
    sources = []
    # Fixed reviewed calibration subset. Never infer shape labels from colour.
    for index in range(1, 6):
        folder = Path(args.results_root) / 'multi-02' / f'frame-{index:02d}'
        results = json.loads((folder / 'results-v2.json').read_text(encoding='utf-8'))
        for item in results:
            path = folder / item['crop_image']
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'unreadable crop: {path}')
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = ((hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 25)).astype(np.uint8) * 255
            mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=2)
            if np.count_nonzero(mask) < 25:
                raise ValueError(f'insufficient interior: {path}')
            hue = float(np.median(hsv[:, :, 0][mask > 0]))
            label = 'blue' if hue >= 100 else 'cyan'
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            value = np.asarray(cv2.mean(lab, mask=mask)[:3])
            samples[label].append(value)
            sources.append({'file': str(path), 'label': label, 'lab': value.tolist()})
    config = yaml.safe_load(Path(args.base_config).read_text(encoding='utf-8'))
    for label, values in samples.items():
        if len(values) < 3:
            raise ValueError('at least three reviewed samples per colour required')
        prototype = np.round(np.median(values, axis=0)).astype(np.uint8)
        rgb = cv2.cvtColor(prototype.reshape(1, 1, 3), cv2.COLOR_LAB2RGB)[0, 0]
        config['classification']['colors'][label]['hex'] = '#' + ''.join(f'{v:02x}' for v in rgb)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    report = {'calibration_only': True, 'calibration_frames': 'multi-02/frame-01..05',
              'unvalidated_colors': ['red', 'yellow', 'green', 'black'],
              'samples': sources, 'colors': config['classification']['colors']}
    target.with_suffix('.calibration.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['colors'], ensure_ascii=False))


if __name__ == '__main__':
    main()
