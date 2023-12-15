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
        self.sensitive_attributes = torch.vstack(
            (religion, race, gender_and_sexual_orientation)).T
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

        num_a_values = sensitive_attributes.shape[1]

        group_fpr_dict = {}
        group_tpr_dict = {}
        group_fnr_dict = {}
        balanced_accuracy = {}

        # use loop to compute
        y = 1
        for a_idx in range(num_a_values):
            group_id = sensitive_attributes[:, a_idx] == 1
            group_com_id = sensitive_attributes[:, a_idx] == 0

            # Calculate the total positive/negative examples for the sensitive group
            group_pos_label_ids = (references == y) & (group_id)
            group_neg_label_ids = (references != y) & (group_id)
            group_pos = torch.sum(group_pos_label_ids) if torch.sum(
                group_pos_label_ids) != 0 else torch.tensor(0)
            group_neg = torch.sum(group_neg_label_ids) if torch.sum(
                group_neg_label_ids) != 0 else torch.tensor(0)

            # Calculate the total positive/negative examples for the complementary group
            group_pos_label_ids_com = (references == y) & (group_com_id)
            group_neg_label_ids_com = (references != y) & (group_com_id)
            group_pos_com = torch.sum(group_pos_label_ids_com) if torch.sum(
                group_pos_label_ids_com) != 0 else torch.tensor(0)
            group_neg_com = torch.sum(group_neg_label_ids_com) if torch.sum(
                group_neg_label_ids_com) != 0 else torch.tensor(0)

            # Calculate TP/FP for the sensitive group
            group_tp_ids = (references == y) & (predictions == y) & (group_id)
            group_fp_ids = (references != y) & (predictions == y) & (group_id)
            group_fn_ids = (references == y) & (predictions != y) & (group_id)

            # Calculate TP/FP for the complementary group
            group_tp_ids_com = (references == y) & (predictions == y) & (group_com_id)
            group_fp_ids_com = (references != y) & (predictions == y) & (group_com_id)
            group_fn_ids_com = (references == y) & (predictions != y) & (group_com_id)

            group_tpr = torch.sum(group_tp_ids) / group_pos if group_pos != 0 else torch.tensor(0)
            group_fpr = torch.sum(group_fp_ids) / group_neg if group_neg != 0 else torch.tensor(0)
            group_fnr = torch.sum(group_fn_ids) / group_pos if group_pos != 0 else torch.tensor(0)
            group_tpr_com = torch.sum(group_tp_ids_com) / group_pos_com if group_pos_com != 0 else torch.tensor(0)
            group_fpr_com = torch.sum(group_fp_ids_com) / group_neg_com if group_neg_com != 0 else torch.tensor(0)
            group_fnr_com = torch.sum(group_fn_ids_com) / group_pos_com if group_pos_com != 0 else torch.tensor(0)

            group_tpr_dict[(1, a_idx)] = group_tpr
            group_fpr_dict[(1, a_idx)] = group_fpr
            group_fnr_dict[(1, a_idx)] = group_fnr
            group_tpr_dict[(0, a_idx)] = group_tpr_com
            group_fpr_dict[(0, a_idx)] = group_fpr_com
            group_fnr_dict[(0, a_idx)] = group_fnr_com
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
            'fnr': group_fnr_dict,
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
        num_a_values = torch.tensor(sensitive_attributes).shape[1]

        # EO Calculation
        group_metric_rates = self.compute_deo_for_multi_label_loop(
            predictions=predictions,
            references=references,
            sensitive_attributes=sensitive_attributes,
        )

        eo_per_attribute = []
        fnr_dict = {}
        for a_idx in range(num_a_values):
            tpr_diff = torch.abs(group_metric_rates['tpr'][(
                1, a_idx)] - group_metric_rates['tpr'][(0, a_idx)])
            fpr_diff = torch.abs(group_metric_rates['fpr'][(
                1, a_idx)] - group_metric_rates['fpr'][(0, a_idx)])
            fnr_diff = torch.abs(group_metric_rates['fnr'][(
                1, a_idx)] - group_metric_rates['fnr'][(0, a_idx)])
            fnr_dict["fnr_diff/" + self.mapping[a_idx]] = fnr_diff
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
                new_key = key + "/" + self.mapping[subkey[1]] + groupname
                unpacked_dict[new_key] = subvalue

        return {**unpacked_dict, **metrics, **fnr_dict}
    
    
P_MAPPING_INDEX_CLASS = {
    0: 'professor',
    1: 'physician',
    2: 'attorney',
    3: 'photographer',
    4: 'journalist',
    5: 'nurse',
    6: 'psychologist',
    7: 'teacher',
    8: 'dentist',
    9: 'surgeon',
    10: 'architect',
    11: 'painter',
    12: 'model',
    13: 'poet',
    14: 'filmmaker',
    15: 'software_engineer',
    16: 'accountant',
    17: 'composer',
    18: 'dietitian',
    19: 'comedian',
    20: 'chiropractor',
    21: 'pastor',
    22: 'paralegal',
    23: 'yoga_teacher',
    24: 'dj',
    25: 'interior_designer',
    26: 'personal_trainer',
    27: 'rapper'
}
P_MAPPING_CLASS_INDEX = {v: k for k, v in P_MAPPING_INDEX_CLASS.items()}


G_MAPPIN_INDEX_CLASS = {0: 'f', 1: 'm'}
G_MAPPING_CLASS_INDEX = {v: k for k, v in G_MAPPIN_INDEX_CLASS.items()}

class TPRGap(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)
        self.add_state("gender", default=[], dist_reduce_fx=None)

    def update(self, preds, target, gender):
        self.gender = gender
        self.preds = preds
        self.target = target

    def compute(self):
        return self._compute(self.preds, self.target, self.gender)

    def _get_unique_labels(self, labels):
        unique_labels = torch.unique(labels)
        return unique_labels.tolist()

    def _compute(self, predictions, label, gender):
        ''' Actual Implementation of compute metrics
        '''

        # Assuming that each row in the logits corresponds to a sample and contains probabilities for each class
        # You can use argmax to get the predicted class for each sample
        predicted_labels = torch.argmax(predictions, dim=1)

        # Initialize dictionaries to store true positives and false negatives per gender and label class
        tp_dict = {}
        fn_dict = {}

        # Iterate through each gender and label class combination
        for g in torch.unique(gender):
            for l in torch.unique(label):
                mask = (gender == g) & (label == l)
                mask = mask.squeeze()
                true_positives = torch.sum((predicted_labels[mask] == l).float())
                false_negatives = torch.sum((predicted_labels[mask] != l).float())
                tp_dict[(G_MAPPIN_INDEX_CLASS[g.item()], P_MAPPING_INDEX_CLASS[l.item()])] = true_positives
                fn_dict[(G_MAPPIN_INDEX_CLASS[g.item()], P_MAPPING_INDEX_CLASS[l.item()])] = false_negatives

        # Calculate True Positive Rate (TPR) per gender and label class
        tpr_dict = {key: tp_dict[key] / (tp_dict[key] + fn_dict[key]) for key in tp_dict}

        # Calculate and print the difference of TPR
        for l in torch.unique(label):
            tpr_diff = tpr_dict[('f', P_MAPPING_INDEX_CLASS[l.item()])] \
                - tpr_dict[('m', P_MAPPING_INDEX_CLASS[l.item()])]

        unified_dict = {}
        for (g, l), tpr in tpr_dict.items():
            unified_dict[f"TPR_{g}_{l}"] = tpr.item()

        # Calculate and save the difference of TPR
        tpr_diff_values = []
        for l in torch.unique(label):
            tpr_diff = tpr_dict[('f', P_MAPPING_INDEX_CLASS[l.item()])] - tpr_dict[('m', P_MAPPING_INDEX_CLASS[l.item()])]
            tpr_diff_values.append(tpr_diff)
            unified_dict[f"TPR_Diff_{P_MAPPING_INDEX_CLASS[l.item()]}"] = tpr_diff.item()
            
        rmse = torch.sqrt(torch.mean(torch.tensor(tpr_diff_values) ** 2))
        
        unified_dict["TPR_RMSE"] =  rmse
        
        # Calculate TPED
        TPED = sum(abs(tpr_dict[('f', P_MAPPING_INDEX_CLASS[l.item()])] \
                       - tpr_dict[('m', P_MAPPING_INDEX_CLASS[l.item()])]) for l in torch.unique(label))

            
        return {**unified_dict}
    
