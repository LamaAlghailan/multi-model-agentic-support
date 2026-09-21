import collections
import re
import string

def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def qa_scores(prediction, reference):
    p, r = normalize(prediction), normalize(reference)
    pt, rt = p.split(), r.split()
    common = sum((collections.Counter(pt) & collections.Counter(rt)).values())
    f1 = 2 * common / (len(pt) + len(rt)) if pt or rt else 1.0
    return {'exact_match': float(p == r), 'token_f1': f1}

def rouge_l(prediction, reference):
    """Whitespace-token ROUGE-L F1, without stemming; consistently used before/after."""
    p, r = prediction.lower().split(), reference.lower().split()
    row = [0] * (len(r) + 1)
    for a in p:
        previous = row[:]
        for j, b in enumerate(r, 1):
            row[j] = previous[j-1] + 1 if a == b else max(previous[j], row[j-1])
    return 2 * row[-1] / (len(p) + len(r)) if p or r else 1.0

def classification(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    _, per_class, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {'accuracy': float(accuracy_score(y_true, y_pred)), 'precision_macro': float(precision),
            'recall_macro': float(recall), 'f1_macro': float(f1),
            'per_class_recall': {str(k): float(v) for k, v in zip(labels, per_class)},
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist()}

def threshold_sweep(truth, predicted, confidences, target_accuracy=0.9, minimum_coverage=0.5):
    """Validation only. No qualifying policy means abstain from classifier routing."""
    rows = []
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        accepted = [i for i, c in enumerate(confidences) if c >= threshold]
        accuracy = sum(truth[i] == predicted[i] for i in accepted) / len(accepted) if accepted else None
        rows.append({'threshold': threshold, 'coverage': len(accepted)/len(truth), 'accepted_accuracy': accuracy,
                     'accepted_examples': len(accepted)})
    eligible = [r for r in rows if r['coverage'] >= minimum_coverage and r['accepted_accuracy'] is not None and r['accepted_accuracy'] >= target_accuracy]
    best = max(eligible, key=lambda r: (r['coverage'], r['accepted_accuracy'], r['threshold'])) if eligible else None
    return {'selected_threshold': best['threshold'] if best else 1.0, 'policy_qualified': best is not None, 'rows': rows,
            'selection_split': 'validation', 'target_accuracy': target_accuracy, 'minimum_coverage': minimum_coverage}
