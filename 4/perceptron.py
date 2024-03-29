from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def to_prob(preds_score):
    preds_prob = []
    up = float(max(preds_score))
    bottom = float(min(preds_score))
    for item in preds_score:
        if item > 0:
            preds_prob.append(item / up)
        else:
            preds_prob.append(0.5 - item / bottom)
    return preds_prob


def predict(weights, xs):
    result = 0
    for x in xs:
        if x in weights:
            result += weights[x]
    y_pred = 1 if result > 0 else -1
    return y_pred, result


def misclassify(weights, xs, y):
    result = 0
    for x in xs:
        if x in weights:
            result += weights[x]
    result *= y
    return result <= 0


def parse(line):
    data = line.rstrip().split(' ')
    y = int(data[0])
    xs = []
    for item in data[1:]:
        xs.append(int(item.split(':')[0]))
    return xs, y


weights = {}
with open('./data/train.dat') as f:
    # All the training data can be only processed once, just like a stream input
    for line in f:
        xs, y = parse(line)
        if misclassify(weights, xs, y) is True:
            for x in xs:
                if x in weights:
                    weights[x] += y
                else:
                    weights[x] = y

tests = []
preds = []
preds_score = []

with open('./data/test.dat') as f:
    for line in f:
        xs, y = parse(line)
        tests.append(y)
        y_pred, result = predict(weights, xs)
        preds.append(y_pred)
        preds_score.append(result)

# Final precision, recall and F1 score on the testing dataset.
print(classification_report(tests, preds))

# ROC curve (as image) on the testing dataset.
fpr, tpr, threshold = metrics.roc_curve(tests, to_prob(preds_score))
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
