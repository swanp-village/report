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
        src = json.dumps(config, indent=4)
        self.target.joinpath('config.json').write_text(src)

    def generate_image_path(self):
        return self.target.joinpath('out.pdf')

    def save_data_as_csv(self, x, y):
        path = self.target.joinpath('out.csv')
        with open(path, 'w') as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter='\t')
            tsv_writer.writerows(zip(x.tolist(), y.tolist()))
