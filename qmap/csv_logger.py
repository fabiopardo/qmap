from datetime import datetime
import os


class CSVLogger:
    def __init__(self, labels, path):
        self.n = len(labels)
        self.path = path + '.csv'
        assert not os.path.exists(self.path), self.path
        print('[CSVLogger] logging {} in {}'.format(labels, self.path))
        assert not os.path.exists(path)
        directory_path = os.path.dirname(self.path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(self.path, 'a') as csv_file:
            csv_file.write(','.join(labels) + '\n')
        self.steps = 0

    def log(self, *values):
        assert len(values) == self.n
        with open(self.path, 'a') as csv_file:
            csv_file.write(','.join([str(v) for v in values]) + '\n')
