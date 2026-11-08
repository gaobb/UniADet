import torch
from .metrics import AUPR, AUPRO, AUROC, F1Max

class Evaluator(object):
    def __init__(self, device, metrics=[]):
        if len(metrics) == 0:
            self.metrics = [
                'I-AUROC', 'I-AP', 'I-F1max', 
                'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO'
            ]
        else:
            self.metrics = metrics

        self.aupr = AUPR().to(device)
        self.aupro = AUPRO().to(device)
        self.auroc = AUROC().to(device)
        self.f1max = F1Max().to(device)

    def run(self, results, cls_name, logger=None):
        self.auroc.reset()
        self.aupr.reset()
        self.aupro.reset()
        self.f1max.reset()
        
        idxes = results['cls_names'] == cls_name

        gt_px = results['gt_masks'][idxes] 
        pr_px = results['pr_masks'][idxes]

        gt_sp = results['gt_anomalys'][idxes]
        pr_sp = results['pr_anomalys'][idxes]
      
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)

        eval_results = {}
        for metric in self.metrics:
            if metric.startswith('I-AUROC'):
                eval_results[metric] = self.auroc(pr_sp, gt_sp).item()
                
            elif metric.startswith('P-AUROC'):
                eval_results[metric] = self.auroc(pr_px.ravel(), gt_px.ravel()).item()
                    
            elif metric.startswith('I-AP'):
                eval_results[metric] = self.aupr(pr_sp, gt_sp).item()
                
            elif metric.startswith('P-AP'):
                eval_results[metric] = self.aupr(pr_px.ravel(), gt_px.ravel()).item()
                  
            elif metric.startswith('I-F1max'):
                eval_results[metric] = self.f1max(pr_sp, gt_sp).item()
            
            elif metric.startswith('P-F1max'):
                eval_results[metric] = self.f1max(pr_px.ravel(), gt_px.ravel()).item()
            
            elif metric.startswith('P-AUPRO'):
                eval_results[metric] = self.aupro(pr_px, gt_px).item()
         
        return eval_results