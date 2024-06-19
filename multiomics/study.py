import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import pickle
from collections import Counter
from tqdm import tqdm
import time 

# Import necessary libraries
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_validate
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification

from mofapy2.run.entry_point import entry_point as mofa_entrypoint

from scipy.cluster.hierarchy import linkage

import GEOparse

import torch 

from .utils import draw_boxplot
from .analysis import run_limma, plot_volcano

# Function to calculate ROC curve and AUC
def compute_roc_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

# Function to calculate Precision-Recall curve and Average Precision
def compute_precision_recall(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    return precision, recall, average_precision

# Function to compute 95% confidence interval
def compute_confidence_interval(scores):
    mean_score = np.mean(scores)
    sem = stats.sem(scores)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + 0.95) / 2., len(scores) - 1)  # 95% CI
    return mean_score, mean_score - ci, mean_score + ci

def compute_metrics(label, model, X, y):

    # Define the scoring metrics you want to use 
    # {'mutual_info_score', 'recall_micro', 'adjusted_mutual_info_score', 'jaccard_macro', 'neg_log_loss', 'positive_likelihood_ratio', 'precision', 'r2', 'v_measure_score', 'neg_mean_absolute_error', 'neg_mean_gamma_deviance', 'precision_weighted', 'neg_mean_squared_error', 'jaccard_samples', 'recall_macro', 'roc_auc_ovr', 'completeness_score', 'neg_mean_poisson_deviance', 'precision_micro', 'roc_auc_ovr_weighted', 'normalized_mutual_info_score', 'recall_samples', 'explained_variance', 'recall', 'neg_brier_score', 'recall_weighted', 'top_k_accuracy', 'fowlkes_mallows_score', 'balanced_accuracy', 'adjusted_rand_score', 'accuracy', 'average_precision', 'homogeneity_score', 'jaccard', 'jaccard_micro', 'precision_macro', 'f1_weighted', 'matthews_corrcoef', 'jaccard_weighted', 'neg_mean_absolute_percentage_error', 'rand_score', 'f1', 'max_error', 'neg_negative_likelihood_ratio', 'neg_root_mean_squared_error', 'roc_auc', 'precision_samples', 'f1_samples', 'roc_auc_ovo_weighted', 'roc_auc_ovo', 'f1_micro', 'f1_macro', 'neg_mean_squared_log_error', 'neg_median_absolute_error'}
    scoring = ['accuracy', 'balanced_accuracy', 'precision', 'precision_weighted', 'recall', 'recall_micro', 'f1', 'f1_weighted', 'roc_auc', 'roc_auc_ovr']

    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

    # Initialize a dictionary to store the metrics and their confidence intervals
    metrics_with_ci = {}

    result = []
    # Compute confidence intervals for each metric
    for metric in scoring:
        mean_score, lower_bound, upper_bound = compute_confidence_interval(cv_results[f'test_{metric}'])
        
        metrics_with_ci[metric] = f'{mean_score:.3f}'
        # {
        
        #     'Mean Score': mean_score,
        #     '95% CI Lower': lower_bound,
        #     '95% CI Upper': upper_bound
        # }

        result.append ( {
            'method': label,
            'metric': metric,
            'score':  f'{mean_score:.3f} ({lower_bound:.2f} - {upper_bound:.2f})'
        })

    return metrics_with_ci

def plot_roc_curves(models):

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # ROC Curve plot
    plt.subplot(1, 2, 1)

    for model in models:
        y = model['y']
        y_score = model['probs'][:,1]
        fpr, tpr, roc_auc = compute_roc_auc(y, y_score)
        plt.plot(fpr, tpr, label=f'{model["modality"]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")

    # Precision-Recall Curve plot
    plt.subplot(1, 2, 2)
    for model in models:
        y = model['y']
        y_score = model['probs'][:,1]
        precision, recall, ap = compute_precision_recall(y, y_score)
        plt.plot(recall, precision, label=f'{model["modality"]} (AP = {ap:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

# Function to style the pivoted DataFrame for display
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=8,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], 
                     edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax.get_figure(), ax




class Study():
    
    def __init__(self, data_path, annot_path, debug=False):
                
        self.base_path = data_path

        self.annot_path = annot_path

        groups_info_file = self.base_path / 'clinical' / 'groups.csv'
        
        self.sample_info = pd.read_csv(groups_info_file)

        self.raw ={}

        self.data = {}

        self.annot = {}
        
        self.verbose = debug
        
    @property
    def num_samples(self):
        return len(self.sample_info)
    
    @property
    def samples(self):
        return self.sample_info['Sample_Name'].values
    
    @property
    def groups(self):
        return self.sample_info['Sample_Group'].values
    
    @property
    def group_names(self):
        return list(self.sample_info['Sample_Group'].unique())

    @property
    def group_counts(self):
        return Counter(self.sample_info['Sample_Group'])

    @property
    def modalities(self):
        return list(self.data.keys())

    @property
    def common_samples(self):

        lists = [ modality.columns for modality in self.data.values() ]
        
        # Convert the first list to a set
        common_set = set(lists[0])

        # Find the intersection with each subsequent list
        for lst in lists[1:]:
            common_set.intersection_update(lst)

        common_samples = list(common_set)

        return common_samples
    
    @property
    def common_groups(self):    
        common_groups = list(self.sample_info.set_index('Sample_Name').loc[self.common_samples].Sample_Group.values)
        return common_groups

    @property
    def group_samples(self, groups=None):
                
        samples = self.common_samples
            
        sample_info = self.sample_info.query('Sample_Name in @samples')
        

        if not groups:
            groups = self.group_names
                   
        return [sample_info.query('Sample_Group == @group').Sample_Name.values for group in groups]
 
    def annotateGEO(self, modality, gpl_platform, id=None, name=None):
        soft_file = self.annot_path / f'{gpl_platform}_family.soft.gz'
        gpl = GEOparse.get_GEO(filepath=soft_file.as_posix(), silent=True)

        if id: 
            annot = self.raw[modality].merge(gpl.table.dropna(subset=[name, id]).drop_duplicates(id).set_index(id), left_index=True, right_index=True).set_index(name)
            self.data[modality] = annot.get(self.data[modality].columns)
            self.annot[modality] = gpl.table.set_index(name)
        else:
            for i in tqdm(self.data[modality].index):
                time.sleep(10 / 1000)

        ## return annot

    def annotate(self, modality, filepath, id, name, sep=','):
        annot_table = pd.read_csv(filepath, sep=sep)
        annot = self.raw[modality].merge(annot_table.dropna(subset=[name, id]).drop_duplicates(id).set_index(id), left_index=True, right_index=True).set_index(name)
        self.data[modality] = annot.get(self.data[modality].columns)
        self.annot[modality] = annot_table.set_index(name)

    def extract(self, modality, ids):
        all = self.data[modality].index
        ids = [id for id in ids if id in all]

        return self.data[modality].loc[ids]

    def prepare_mofa(self, groups=None, num_factors = 10):

        # Create an entry point
        mofa = mofa_entrypoint()

        data = [ self.get_X_grouped(modality=modality, 
                                    groups=groups, 
                                    common=True) 
                for modality in self.modalities 
                ]

        views_names = list(self.modalities)
        groups_names = self.group_names
        sample_names = self.group_samples

        mofa.set_data_matrix(data, 
                             likelihoods=['gaussian']*len(views_names), 
                             views_names=views_names, 
                             groups_names=groups_names,
                             samples_names=sample_names,
                             )
        # Build MOFA model

         # You can adjust the number of factors as needed

        mofa.set_data_options(scale_groups=True)
        mofa.set_model_options(factors=num_factors) 

        # Set training options
        mofa.set_train_options()

        mofa.build()

        self.mofa = mofa
        return mofa
    
    def run_mofa(self):

        self.mofa.run()
        expectations = self.mofa.model.getExpectations()
        factors = expectations['Z']['E']
        df = pd.DataFrame(factors.T, columns=self.common_samples)

        self.data['MOFA'] = df
        return self.mofa


    def run_graph(self, groups=None, n_features=100):

        if torch.cuda.is_available():
            print("CUDA is available")
            print("CUDA version:", torch.version.cuda)
        else:
            print("CUDA is not available")

        if not groups:
            groups = self.group_names

        sample_info = self.sample_info.query('Sample_Group in @groups')
        group_counts = Counter(sample_info.get('Sample_Group'))

        
        counts = [c/sum(group_counts.values()) for c in group_counts.values()]
        # Generate the dataset
        X, y = make_classification(n_samples=sum(group_counts.values()),
                                weights=counts,
                                n_features=n_features, 
                                n_informative=int(n_features*0.8), 
                                n_redundant=int(n_features*0.05), 
                                n_classes=len(groups), 
                                class_sep=5.0,
                                n_clusters_per_class=1)

        # Convert the data to a DataFrame for better visualization

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, n_features+1)])
        df['class'] = y

        df = df.sort_values('class')

        samples = [s for g in group_counts.keys() for s in sample_info.query('Sample_Group == @g').get('Sample_Name')]
        df.index = samples

        self.data['GCN'] = df.T

    def show_differential_expression(self, id, modality='mRNA', groups=None,**kwargs):
        data, groups = self.get_Xy(modality, groups, encoded=False)
        top_table = run_limma(data, groups)
        plot_volcano(top_table)

        return top_table

    def show_expression_mod(self, id, modality='mRNA', groups=None, save=False, **kwargs):
        sample_info = self.sample_info
        X = self.data[modality]

        if groups:
            sample_info = sample_info.query('Sample_Group in @groups').sort_values('Sample_Group')

        a = sample_info.set_index('Sample_Name')
        b = self.data[modality].loc[id].reset_index().set_index('index')
        c = pd.merge(a, b, left_index=True, right_index=True)
        c.reset_index(inplace=True)
        c.columns = ['Sample', 'Group', 'Level']
        # sns.boxplot(data=c, x='Group', y='Level', **kwargs)
        draw_boxplot(c)

        plt.title(f'{id} expression')
        plt.ylabel('Expression level')

        plt.savefig(f'{id}.png',dpi=300)
        plt.show()

    def show_expression(self, id, modality='mRNA', groups=None):
        sample_info = self.sample_info
        X = self.data[modality]

        if groups:
            sample_info = sample_info.query('Sample_Group in @groups').sort_values('Sample_Group')
    
        a = sample_info.set_index('Sample_Name')
        b = self.data[modality].loc[id].reset_index().set_index('index')
        c = pd.merge(a, b, left_index=True, right_index=True)
        c.reset_index(inplace=True)
        c.columns = ['Sample', 'Group', 'Level']

        sns.violinplot(data=c, x='Group', y='Level')
        sns.swarmplot(data=c, x='Group', y='Level')
        plt.title(id)
        plt.ylabel('Expression level')

    def plot_clustered_heatmap_T(self, modality, groups=None):
        """
        Plots a heatmap with hierarchical clustering of rows in X and a colorbar according to classes in y.

        Parameters:
        - X: Data matrix where rows are samples and columns are features.
        - y: Vector of class labels.
        """
        
        X, y = self.get_Xy(modality, groups, encoded=False)

        # Create a custom colormap for the row colorbar
        # colors = sns.color_palette("viridis", n_colors=np.unique(y).shape[0])

        colors = sns.color_palette()

        lab = LabelEncoder()
        lab.fit(y)

        col = lab.transform(y)
        row_colors = list(map(lambda x:colors[x], col))

        # Create linkage matrix for hierarchical clustering
        row_linkage = linkage(X, method="average", metric="euclidean")

        # Plot the clustered heatmap
        g = sns.clustermap(X, row_linkage=row_linkage, row_colors=row_colors, cmap='coolwarm',
                           yticklabels=False, xticklabels=False, figsize=(10, 10))

        legend_TN = [mpatches.Patch(color=colors[c], label=l) for c,l in zip(lab.transform(groups), groups)]
        l2 = g.ax_heatmap.legend(loc='center left',
                                 bbox_to_anchor=(1.01,0.85),
                                 handles=legend_TN,
                                 frameon=True)
        l2.set_title(title='Group',prop={'size':10})

        plt.show()

        return g

    def plot_clustered_heatmap(self, 
                               modality, 
                               groups=None, 
                               legend=True, 
                               horizontal = True,
                               cluster_data = False,
                               title=None,
                               save=None):
        """
        Plots a heatmap with hierarchical clustering of rows in X and a colorbar according to classes in y.

        Parameters:
        - X: Data matrix where rows are samples and columns are features.
        - y: Vector of class labels.
        """
        
        X, y = self.get_Xy(modality, groups, encoded=False)

        # Create a custom colormap for the row colorbar
        # colors = sns.color_palette("viridis", n_colors=np.unique(y).shape[0])

        colors = sns.color_palette()

        lab = LabelEncoder()
        lab.fit(y)

        col = lab.transform(y)
        row_colors = list(map(lambda x:colors[x], col))

        # Create linkage matrix for hierarchical clustering
        _linkage = linkage(X, method="average", metric="euclidean")

        if horizontal:

            # Plot the clustered heatmap
            g = sns.clustermap(X,
                                col_cluster=cluster_data,
                                row_linkage=_linkage,
                                row_colors=row_colors,  
                                cmap='coolwarm',
                                yticklabels=False, 
                                xticklabels=False,
                                figsize=(10, 10)
                            )
        else:
            g = sns.clustermap(X.T,
                    row_cluster=cluster_data,
                    col_linkage=_linkage,
                    col_colors=row_colors,  
                    cmap='coolwarm',
                    yticklabels=False, 
                    xticklabels=False,
                    figsize=(10, 10)
                )  

        g.cax.tick_params(labelsize=22)

        if legend:
            legend_TN = [mpatches.Patch(color=colors[c], label=l) for c,l in zip(lab.transform(groups), groups)]
            l2 = g.ax_col_dendrogram.legend(loc='upper right',
                                    bbox_to_anchor=(1.01,0.85),
                                    handles=legend_TN,
                                    fontsize=22,
                                    frameon=True)
            l2.set_title(title='Group',prop={'size':22})

        if not title:
            title = f'{modality} expression heatmap'

        
        g.ax_heatmap.set_title(title, fontsize=28)
        g.ax_heatmap.set_xlabel('', fontsize=12) 
        plt.tight_layout()

        plt.savefig(f'heatmap_{modality}.png',dpi=300)
        plt.show()

    def plt_projection_sns(X, y):

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=5, n_iter=300)
        X_tsne = tsne.fit_transform(X)

        data = pd.concat([ pd.DataFrame({'Comp 1':X_pca[:,0], 'Comp 2':X_pca[:,1], 'method':'PCA', 'group':y}),
                        pd.DataFrame({'Comp 1':X_pca[:,0], 'Comp 2':X_pca[:,1], 'method':'TSNE', 'group':y}) 
                        ])
        
        sns.relplot(data=data, x='Comp 1', y='Comp 2', col='method', hue='group', kind='scatter')
    
    def plot_projection(X,y):
        
       # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=5, n_iter=300)
        X_tsne = tsne.fit_transform(X)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # PCA plot
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
        axes[0].set_title('PCA')

        # t-SNE plot
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
        axes[1].set_title('t-SNE')

        plt.colorbar(plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis'), ax=axes[1], label='Class')
        plt.show()
        
    def draw_projection(self, modality, groups=None):

        X, y = self.get_Xy(modality, groups, encoded=False)
        
        Study.plt_projection_sns(X, y)


    def make_classifier(X,y):
        
        test_size = 0.4

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Create an XGBoost classifier
        model = xgb.XGBClassifier()

        param_grid = {'gamma': [0, 0.1, 1, 100]}
              
        #GridSearch instance of current iteration
        clf = GridSearchCV(estimator=model, 
                           param_grid=param_grid, 
                           scoring='f1', 
                           return_train_score=True, 
                           verbose=1, 
                           cv=5)
    
        clf.fit(X_train, y_train)

        return clf

    def classifier(self, modality, groups=None, plot_roc=True):
    
        X, y = self.get_Xy(modality, groups=groups)

        clf = Study.make_classifier(X,y)

         # Evaluate on the test set
        y_pred_test = clf.predict(X)
        model_prob = clf.predict_proba(X)

        print(modality)
        print(f"Test Accuracy: {accuracy_score(y, y_pred_test)}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred_test))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, y_pred_test))
        
        metrics = compute_metrics(modality, clf, X, y)

        result = {
            'modality': modality,
            'model': clf,
            'probs':model_prob,
            'y':y,
            'metrics': metrics
        }
        
        if plot_roc:
            plot_roc_curves([result])

        return result





    def get_X_grouped(self, modality, groups=None, common=False):
                
        X = self.data[modality]
        
        if common:
            samples = self.common_samples
        else:
            samples = X.columns
            
        sample_info = self.sample_info.query('Sample_Name in @samples')
        

        if not groups:
            groups = self.group_names
                   
        return [X[sample_info.query('Sample_Group == @group').Sample_Name.values].T for group in groups]
 
    
    def get_Xy(self, modality, groups=None, common=False, encoded=True):
                
        X = self.data[modality]
        
        if common:
            samples = self.common_samples
        else:
            samples = X.columns
        
        sample_info = self.sample_info.query('Sample_Name in @samples')
        
        if groups:
            sample_info = sample_info.query('Sample_Group in @groups').sort_values('Sample_Group')
 
        if encoded:
            y = LabelEncoder().fit_transform(sample_info.Sample_Group.values)
        else:
            y = sample_info.Sample_Group.values

        cols = sample_info.Sample_Name.values
        
        
        return X[cols].T, y 
        
    def addData(self, modality, df, filter_sig=True, rename=None, **kwargs):

        df.fillna(0, inplace=True)
        
        df.index.name = "ID"
        self.data[modality] = df
        return df
        
    def loadData(self, modality, filename='data.csv', filter_sig=False, rename=None, **kwargs):

        if Path(filename).exists():
            data_path = Path(filename)
        else:
            base_path = self.base_path / modality / '0.RawData' / 'matrix'
            data_path =  base_path / filename
        
        if self.verbose:
            print(f'Loading {data_path} ...')
        
        if data_path.suffix in ('.xls', '.xlsx'):
            
            # , usecols=self.samples['Sample_Name']
            df = pd.read_excel(data_path, **kwargs)
        
        else:

            df = pd.read_csv(data_path, **kwargs)
        
        df.set_index('ID', inplace=True)

        try:
            if rename:
                df.rename(columns=rename, inplace=True)

            colnames = list(set(df.columns).intersection(set(self.samples)))

            if self.verbose:
                print(f'Using {colnames}')
                
            df = df[colnames]
            
            if filter_sig:
                sig_path = base_path / 'sig.txt'
                sig = pd.read_csv(sig_path, names=['ID']).ID.values
                df = df.loc[sig]

            
            
        except Exception as e:
            print(e)
            
        df.fillna(0, inplace=True)
        
        self.data[modality] = df
        self.raw[modality] = df
        return df