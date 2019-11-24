import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import r2_score

 
def graph_epoch(history):
	print("Printing graph1")
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.savefig('../results/Loss_Graph.png')
	plt.clf()

	print("Printing graph2")
	plt.plot(history.history['det_coeff'])
	plt.plot(history.history['val_det_coeff'])
	plt.title('Model Deterministic Coefficient')
	plt.ylabel('Det Coeff')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.savefig('../results/Det_Graph.png')
	plt.clf()


def pred_test_visual(Y_pred, Y_test, scores):	
	x=[[item[i] for item in Y_pred] for i in range(len(Y_pred[0]))]
	y=[[item[i] for item in Y_test] for i in range(len(Y_test[0]))]

	plt.plot(x[0], y[0], '.', color='blue')
	plt.xlabel('Model Prediction')
	plt.ylabel('RNA-seq')
	plt.text(1,0.03,'$R^2$=%.2f' %(r2_score(y[0], x[0])),fontsize= 13.85,ha='right')
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.savefig('../results/Prediction0.png')
	plt.clf()

	plt.plot(x[1], y[1], '.', color='blue')
	plt.xlabel('Model Prediction')
	plt.ylabel('RNA-seq')
	plt.text(1,0.03,'$R^2$=%.2f' %(r2_score(y[1], x[1])),fontsize= 13.85,ha='right')
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.savefig('../results/Prediction1.png')
	plt.clf()

	plt.plot(x[2], y[2], '.', color='blue')
	plt.xlabel('Model Prediction')
	plt.ylabel('RNA-seq')
	plt.text(1,0.03,'$R^2$=%.2f' %(r2_score(y[2], x[2])),fontsize= 13.85,ha='right')
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.savefig('../results/Prediction2.png')
	plt.clf()

	plt.plot(x[3], y[3], '.', color='blue')
	plt.xlabel('Model Prediction')
	plt.ylabel('RNA-seq')
	plt.text(1,0.03,'$R^2$=%.2f' %(r2_score(y[3], x[3])),fontsize= 13.85,ha='right')
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.savefig('../results/Prediction3.png')
	plt.clf()

	plt.plot(x[4], y[4], '.', color='blue')
	plt.xlabel('Model Prediction')
	plt.ylabel('RNA-seq')
	plt.text(1,0.03,'$R^2$=%.2f' %(r2_score(y[4], x[4])),fontsize= 13.85,ha='right')
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.savefig('../results/Prediction4.png')
	plt.clf()

	print("Deterministic Coefficient: {}".format(scores[1])) 

