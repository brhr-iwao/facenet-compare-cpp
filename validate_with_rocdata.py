# A simple python script to calculate Area under Curve(AUC), Equal Error Rate(EER),
# Threshold at EER and Accuracy.
# This also draws the Reciever Operating Characteristic(ROC) Curve.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(
	description='Caluculates validation scores by using rocdata')
parser.add_argument('rocDataName', type=str, help='The filename of rocdata')
args = parser.parse_args()
data=np.loadtxt(args.rocDataName)
y_true=data[:,1]
y_score=data[:,0]
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# Save ROC curve
plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.savefig(args.rocDataName + '.png')

print('AUC: ',roc_auc_score(y_true, y_score))

fnr=1-tpr
EER=fpr[np.nanargmin(np.absolute((fnr-fpr)))]
eer_threshold=thresholds[np.nanargmin(np.absolute((fnr-fpr)))]
print('EER from false positive rate: ', EER)
EER=fnr[np.nanargmin(np.absolute((fnr-fpr)))]
print('EER from false negative rate: ', EER)
print('Threshold at EER: ', eer_threshold)

y_pred=[]
for i in range(len(y_score)):
	if y_score[i] < eer_threshold:
		y_pred.append(0)
	elif y_score[i] >= eer_threshold:
		y_pred.append(1)

print('Accuracy at EER: ', accuracy_score(y_true,y_pred))

