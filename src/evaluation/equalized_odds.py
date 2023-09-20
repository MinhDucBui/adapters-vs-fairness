from torchmetrics import Metric
import torch
from tqdm import tqdm


class EqualizedOdds(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)
        self.add_state("sensitive_attributes", default=[], dist_reduce_fx=None)
        
    def update(self, preds, target, religion, race, gender_and_sexual_orientation):
        self.sensitive_attributes = torch.vstack((religion, race, gender_and_sexual_orientation)).T
        self.preds = preds
        self.target = target

    def compute(self):
        return self._compute(self.preds, self.target, self.sensitive_attributes)
        
    
    def compute_deo_for_multi_label_loop(self, predictions, references, sensitive_attributes):
        '''compute deo based on loops
        where sensitive_attributes is (multilabel) one-hot encoding
        '''
        assert len(references) != 0
        assert len(predictions) == len(references) == len(sensitive_attributes)

        unique_y_values = torch.unique(torch.vstack((predictions, references))).tolist()
        num_a_values = sensitive_attributes.shape[1]

        overall_fpr_dict, group_fpr_dict = {}, {}
        overall_fnr_dict, group_fnr_dict = {}, {}
        overall_tpr_dict, group_tpr_dict = {}, {}
        overall_tnr_dict, group_tnr_dict = {}, {}

        # use loop to compute
        for y in unique_y_values:

            pos_label_ids = references == y
            neg_label_ids = references != y

            fp_ids = (references != y) & (predictions == y)
            fn_ids = (references == y) & (predictions != y)
            tp_ids = (references == y) & (predictions == y)
            tn_ids = (references != y) & (predictions != y)

            fpr = torch.sum(fp_ids) / torch.sum(neg_label_ids) if torch.sum(neg_label_ids) != 0 else 0
            fnr = torch.sum(fn_ids) / torch.sum(pos_label_ids) if torch.sum(pos_label_ids) != 0 else 0
            tpr = torch.sum(tp_ids) / torch.sum(pos_label_ids) if torch.sum(pos_label_ids) != 0 else 0
            tnr = torch.sum(tn_ids) / torch.sum(neg_label_ids) if torch.sum(neg_label_ids) != 0 else 0

            acc = (torch.sum(tp_ids | tn_ids)) / len(references)

            overall_fpr_dict[y] = fpr
            overall_fnr_dict[y] = fnr
            overall_tpr_dict[y] = tpr
            overall_tnr_dict[y] = tnr

            for a_idx in range(num_a_values):
                print(a_idx)
                group_pos_label_ids = (references == y) & (sensitive_attributes[:, a_idx] == 1)
                group_neg_label_ids = (references != y) & (sensitive_attributes[:, a_idx] == 1)
                group_fp_ids = (references != y) & (predictions == y) & (sensitive_attributes[:, a_idx] == 1)
                group_fn_ids = (references == y) & (predictions != y) & (sensitive_attributes[:, a_idx] == 1)
                group_tp_ids = (references == y) & (predictions == y) & (sensitive_attributes[:, a_idx] == 1)
                group_tn_ids = (references != y) & (predictions != y) & (sensitive_attributes[:, a_idx] == 1)

                group_fpr = torch.sum(group_fp_ids) / torch.sum(group_neg_label_ids) if torch.sum(group_neg_label_ids) != 0 else 0
                group_fnr = torch.sum(group_fn_ids) / torch.sum(group_pos_label_ids) if torch.sum(group_pos_label_ids) != 0 else 0
                group_tpr = torch.sum(group_tp_ids) / torch.sum(group_pos_label_ids) if torch.sum(group_pos_label_ids) != 0 else 0
                group_tnr = torch.sum(group_tn_ids) / torch.sum(group_neg_label_ids) if torch.sum(group_neg_label_ids) != 0 else 0

                # group_acc = torch.sum(group_tp_ids | group_tn_ids).item() / torch.sum(sensitive_attributes[:, a_idx] == 1).item()

                group_fpr_dict[(y, a_idx)] = group_fpr
                group_fnr_dict[(y, a_idx)] = group_fnr
                group_tpr_dict[(y, a_idx)] = group_tpr
                group_tnr_dict[(y, a_idx)] = group_tnr
                
        overall_metric_rates = {
            'overall_fpr_dict': overall_fpr_dict,
            'overall_fnr_dict': overall_fnr_dict,
            'overall_tpr_dict': overall_tpr_dict,
            'overall_tnr_dict': overall_tnr_dict,
        }
        group_metric_rates = {
            'group_fpr_dict': group_fpr_dict,
            'group_fnr_dict': group_fnr_dict,
            'group_tpr_dict': group_tpr_dict,
            'group_tnr_dict': group_tnr_dict,
        }

        return group_metric_rates, overall_metric_rates
    
    def _get_unique_labels(self, labels):
        unique_labels = torch.unique(labels)
        return unique_labels.tolist()
    
    
    def _compute(self, predictions, references, sensitive_attributes):
        ''' Actual Implementation of compute metrics
        '''
        # test whether the references are binary labels
        is_binary_label = len(torch.unique(references)) == 2

        # get cardinality of Y and A
        unique_y_values = torch.unique(torch.vstack((predictions, references))).tolist()
        num_a_values = torch.tensor(sensitive_attributes).shape[1]

        #####################################################
        ######### compute EO-based fairness metrics #########
        #####################################################
        group_metric_rates, overall_metric_rates = self.compute_deo_for_multi_label_loop(
            predictions=predictions,
            references=references,
            sensitive_attributes=sensitive_attributes,
        )

        # by group metrics (could be extended when a is not binary)
        fprs_diff, fnrs_diff = [], []
        tprs_diff, tnrs_diff = [], []
        for y_idx in unique_y_values:
            for a_idx in range(num_a_values):
                fprs_diff.append(torch.abs(group_metric_rates['group_fpr_dict'][(y_idx, a_idx)] - overall_metric_rates['overall_fpr_dict'][y_idx]))
                fnrs_diff.append(torch.abs(group_metric_rates['group_fnr_dict'][(y_idx, a_idx)] - overall_metric_rates['overall_fnr_dict'][y_idx]))
                tprs_diff.append(torch.abs(group_metric_rates['group_tpr_dict'][(y_idx, a_idx)] - overall_metric_rates['overall_tpr_dict'][y_idx]))
                tnrs_diff.append(torch.abs(group_metric_rates['group_tnr_dict'][(y_idx, a_idx)] - overall_metric_rates['overall_tnr_dict'][y_idx]))
        FPR_gap = torch.sum(torch.stack(fprs_diff))
        FNR_gap = torch.sum(torch.stack(fnrs_diff))
        TPR_gap = torch.sum(torch.stack(tprs_diff))
        TNR_gap = torch.sum(torch.stack(tnrs_diff))

        #assert torch.abs(FPR_gap - TNR_gap) <= 1e-6
        #assert torch.abs(FNR_gap - TPR_gap) <= 1e-6 

        # NOTE: by max metrics (the current implementation only works when a is binary)
        #if num_a_values == 2:
            # by max rms
        #    fprs_diff_by_max, fnrs_diff_by_max = [], []
        #    tprs_diff_by_max, tnrs_diff_by_max = [], []
        #    for y_idx in range(len(unique_y_values)):
                # fpr
        #        fpr_diff = group_metric_rates['group_fpr_dict'][(y_idx, 0)] - group_metric_rates['group_fpr_dict'][(y_idx, 1)]
        #        fprs_diff_by_max.append(fpr_diff)
                # fnr
         #       fnr_diff = group_metric_rates['group_fnr_dict'][(y_idx, 0)] - group_metric_rates['group_fnr_dict'][(y_idx, 1)]
         #       fnrs_diff_by_max.append(fnr_diff)
                # tpr
          #      tpr_diff = group_metric_rates['group_tpr_dict'][(y_idx, 0)] - group_metric_rates['group_tpr_dict'][(y_idx, 1)]
         #       tprs_diff_by_max.append(tpr_diff)
                # tnr
         #       tnr_diff = group_metric_rates['group_tnr_dict'][(y_idx, 0)] - group_metric_rates['group_tnr_dict'][(y_idx, 1)]
          #      tnrs_diff_by_max.append(tnr_diff)

          #  fprs_diff_by_max = torch.stack(fprs_diff_by_max)
          #  fnrs_diff_by_max = torch.stack(fnrs_diff_by_max)
          #  tprs_diff_by_max = torch.stack(tprs_diff_by_max)
          #  tnrs_diff_by_max = torch.stack(tnrs_diff_by_max)

          #  assert torch.abs(rms_FNR_gap_by_max - rms_TPR_gap_by_max) <= 1e-6

        metrics = {
            'FPR_gap': FPR_gap,
            'FNR_gap': FNR_gap,
            'EO_gap': FPR_gap + FNR_gap,
        }

        return metrics
