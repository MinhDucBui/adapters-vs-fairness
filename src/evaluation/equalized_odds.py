from torchmetrics import Metric
from torchmetrics.functional.classification import accuracy
import torch


class EqualizedOdds(Metric):
    def __init__(self, attribute_names, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.mapping = {}
        for index, name in enumerate(attribute_names):
            self.mapping[index] = name
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

        group_fpr_dict = {}
        group_tpr_dict = {}
        balanced_accuracy = {}

        # use loop to compute
        y = 1
        for a_idx in range(num_a_values):
            group_id = sensitive_attributes[:, a_idx] == 1

            # Calculate the total positive/negative examples for the sensitive group
            group_pos_label_ids = (references == y) & (group_id)
            group_neg_label_ids = (references != y) & (group_id)
            group_pos = torch.sum(group_pos_label_ids) if torch.sum(group_pos_label_ids) != 0 else 0
            group_neg = torch.sum(group_neg_label_ids) if torch.sum(group_neg_label_ids) != 0 else 0

            # Calculate the total positive/negative examples for the complementary group
            group_pos_label_ids_com = (references == y) & (sensitive_attributes[:, a_idx] == 0)
            group_neg_label_ids_com = (references != y) & (sensitive_attributes[:, a_idx] == 0)
            group_pos_com = torch.sum(group_pos_label_ids_com) if torch.sum(group_pos_label_ids_com) != 0 else 0
            group_neg_com = torch.sum(group_neg_label_ids_com) if torch.sum(group_neg_label_ids_com) != 0 else 0

            # Calculate TP/FP for the sensitive group
            group_tp_ids = (references == y) & (predictions == y) & (group_id)
            group_fp_ids = (references != y) & (predictions == y) & (group_id)

            # Calculate TP/FP for the complementary group
            group_tp_ids_com = (references == y) & (predictions == y) & (sensitive_attributes[:, a_idx] == 0)
            group_fp_ids_com = (references != y) & (predictions == y) & (sensitive_attributes[:, a_idx] == 0)

            group_tpr = torch.sum(group_tp_ids) / group_pos
            group_fpr = torch.sum(group_fp_ids) / group_neg
            group_tpr_com = torch.sum(group_tp_ids_com) / group_pos_com
            group_fpr_com = torch.sum(group_fp_ids_com) / group_neg_com

            group_tpr_dict[(1, a_idx)] = group_tpr
            group_fpr_dict[(1, a_idx)] = group_fpr
            group_tpr_dict[(0, a_idx)] = group_tpr_com
            group_fpr_dict[(0, a_idx)] = group_fpr_com
            if sum(group_id) != 0:
                preds_group = predictions[group_id]
                reference_group = references[group_id]
                balanced_accuracy[(1, a_idx)] = accuracy(preds_group, reference_group, 
                                                    task="multiclass",
                                                    num_classes=2,
                                                    average="macro")
            else:
                balanced_accuracy[(1, a_idx)] = 0

        group_metric_rates = {
            'fpr': group_fpr_dict,
            'tpr': group_tpr_dict,
            "balanced_accuracy": balanced_accuracy
        }

        return group_metric_rates


    def _get_unique_labels(self, labels):
        unique_labels = torch.unique(labels)
        return unique_labels.tolist()


    def _compute(self, predictions, references, sensitive_attributes):
        ''' Actual Implementation of compute metrics
        '''

        # get cardinality of Y and A
        unique_y_values = torch.unique(torch.vstack((predictions, references))).tolist()
        num_a_values = torch.tensor(sensitive_attributes).shape[1]

        # EO Calculation
        group_metric_rates = self.compute_deo_for_multi_label_loop(
            predictions=predictions,
            references=references,
            sensitive_attributes=sensitive_attributes,
        )

        # by group metrics (could be extended when a is not binary)
        fprs_diff, fnrs_diff = [], []
        tprs_diff, tnrs_diff = [], []

        eo_per_attribute = []
        for a_idx in range(num_a_values):
            tpr_diff = torch.abs(group_metric_rates['tpr'][(1, a_idx)] - group_metric_rates['tpr'][(0, a_idx)])
            fpr_diff = torch.abs(group_metric_rates['fpr'][(1, a_idx)] - group_metric_rates['fpr'][(0, a_idx)])
            eo_per_attribute.append(torch.max(tpr_diff, fpr_diff))
            
        # Initialize an empty dictionary
        metrics = {}
        # Iterate through the list and create key-value pairs
        for index, item in enumerate(eo_per_attribute):
            metrics["eo/" + self.mapping[index]] = item
            
        unpacked_dict = {}
        for key, value in group_metric_rates.items():
            for subkey, subvalue in value.items():
                if subkey[0] == 0:
                    groupname = "_comp"
                else:
                    groupname = ""
                new_key = key + groupname +  "_" + self.mapping[subkey[1]]
                unpacked_dict[new_key] = subvalue
        #for key, values in group_metric_rates.items():
        #    metrics[key + "_" + self.mapping[index]] = values
        return {**unpacked_dict, **metrics}
