from pathlib import Path
from datetime import datetime
import json
import csv


class Logger:
    def __init__(self):
        format = '%Y-%m-%d-%H-%M-%S'
        self.result_path = Path.cwd().parent.joinpath('result')
        self.result_path.mkdir(exist_ok=True)
        self.target = self.result_path.joinpath(datetime.now().strftime(format))
        self.target.mkdir()

    def save_config(self, config):
        self.config = config
        src = json.dumps(config, indent=4)
        self.target.joinpath('config.json').write_text(src)

    def generate_image_path(self, name='out'):
        return self.target.joinpath('{}.pdf'.format(name))

    def save_data_as_csv(self, x, y, name='out'):
        path = self.target.joinpath('{}.tsv'.format(name))
        with open(path, 'w') as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter='\t')
            tsv_writer.writerows(zip(x.tolist(), y.tolist()))

    def save_result(self, K, L):
        result = {
            'eta': self.config['eta'],
            'alpha': self.config['alpha'],
            'K': K,
            'L': L,
            'n_eff': self.config['n_eff'],
            'n_g': self.config['n_g'],
            'center_wavelength': self.config['center_wavelength']
        }
        src = json.dumps(result, indent=4)
        self.target.joinpath('result.json').write_text(src)
