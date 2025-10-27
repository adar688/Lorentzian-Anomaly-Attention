from datetime import datetime
from collections import OrderedDict
import pandas as pd
import json

class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_start_time = None
        self.epoch_end_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.epoch_auc = 0
        self.epoch_ap = 0
        self.epoch_new_auc = 0
        self.epoch_new_ap = 0

    def begin_run(self, run):
        run_start_time = datetime.now()
        self.run_start_time = run_start_time.strftime('%Y-%m-%d %H:%M:%S')

        self.run_params = run
        self.run_count += 1

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        epoch_start_time = datetime.now()
        self.epoch_start_time = epoch_start_time.strftime('%Y-%m-%d %H:%M:%S')

        self.epoch_count += 1
        self.epoch_loss = 0

    def end_epoch(self):

        epoch_end_time = datetime.now()
        self.epoch_end_time = epoch_end_time.strftime('%Y-%m-%d %H:%M:%S')

        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['auc'] = self.epoch_auc
        results['ap'] = self.epoch_ap
        results['new_ap'] = self.epoch_new_ap
        results['new_auc'] = self.epoch_new_auc
        results['loss'] = self.epoch_loss
        for k, v in self.run_params._asdict().items():
            results[k] = v

        self.run_data.append(results)
        # df = pd.DataFrame.from_dict(self.run_data, orient='columns')

    def trace_loss(self, loss):
        self.epoch_loss = loss
    
    def trace_auc(self, auc):
        self.epoch_auc = auc
    
    def trace_ap(self, ap):
        self.epoch_ap = ap

    def trace_new_auc(self, new_auc):
        self.epoch_new_auc = new_auc

    def trace_new_ap(self, new_ap):
        self.epoch_new_ap = new_ap

    def save(self, dir, manifold, nout, nhid):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{dir}/{manifold}_{nout}out_{nhid}hid_data.csv')

        with open(f'{dir}/{manifold}_{nout}out_{nhid}hid_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)