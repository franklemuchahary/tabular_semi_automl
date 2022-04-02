import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import re
import pandas as pd
import numpy as np

class ClassificatonModelEvaluation():
    def __init__(self, true_vs_preds_df, threshold=0.5):
        self.true = true_vs_preds_df['Actuals']
        self.threshold = threshold
        
        if self.true.nunique() > 2:
            is_binary = False
        else:
            is_binary = True
        
        if is_binary:
            self.pred = (true_vs_preds_df['Predicted'] > threshold).astype(int)
            self.pred_probab = true_vs_preds_df['Predicted']
        else:
            self.pred = true_vs_preds_df[true_vs_preds_df.columns.difference(['Actuals'])].idxmax(axis=1)
            self.pred_probab = self.pred
            
    def _draw_confusion_matrix(self, true, preds, save_figure_path='confusion_matrix.jpg'):
        conf_matx = metrics.confusion_matrix(true, preds)
        conf_matx = np.round((conf_matx/conf_matx.flatten().sum()) * 100, 2)
        plt.figure(figsize=(5,5))
        sns.heatmap(conf_matx, annot=True,annot_kws={"size": 11},fmt='g', cbar=False)
        plt.xlabel("Predicted Labels", size=12)
        plt.ylabel("True Labels", size=12)
        plt.tight_layout()
    
        if save_figure_path!='':
            plt.savefig(save_figure_path, dpi=200)
        else:
            plt.show()    
        
        return save_figure_path
    
    def _get_classification_report_df(self, true, preds, write_file_path=''):
        threshold = 0.3

        clf_report = metrics.classification_report(true, preds, output_dict=True)

        clf_report_df = round(pd.DataFrame(clf_report).T,2)
        clf_report_df['support'] = clf_report_df['support'].astype(int)
        clf_report_df = clf_report_df.reset_index().rename(columns={'index':"class-label/metric"})
        
        #if write_file_path!='':
        #    clf_report_df.to_csv(write_file_path, index=False)

        return clf_report_df
    
    def evaluate_binary_multiclass_classfication_models(self, save_figure_path):
        clf_report = self._get_classification_report_df(self.true, self.pred)
        conf_matx = self._draw_confusion_matrix(self.true, self.pred, save_figure_path)
        
        return (clf_report, conf_matx)