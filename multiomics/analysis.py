import warnings
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2
from scipy.stats.mstats import zscore
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

def run_limma(df, groups):
    # Enable automatic conversion between Pandas dataframes and R data.frames
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro_dataframe = ro.conversion.py2rpy(df)

    # Load the LIMMA library in R
    limma = importr('limma')

    # Create a design matrix for the linear models
    ro.globalenv['group'] = ro.FactorVector(groups)
    ro.r('design <- model.matrix(~group)')
    
    # Fit the linear model
    ro.globalenv['data'] = ro_dataframe

    ro.r('fit <- lmFit(t(data), design)')

    # Apply empirical Bayes smoothing
    ro.r('fit <- eBayes(fit)')

    # Get the topTable of results (adjust number as needed, here set to retrieve all results)
    top_table = ro.r('topTable(fit, adjust="BH", number=Inf)')

    # Convert the topTable R data.frame back to a Pandas DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        pandas_top_table = ro.conversion.rpy2py(top_table)

    return pandas_top_table

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import networkx as nx 

from sklearn.metrics.pairwise import cosine_similarity

import polars as pl

def patient_similarity(data):

	df = pl.from_pandas(data)

	# Calculate the standard deviation for each row
	std_dev = df.transpose().select(pl.all().std())

	# Convert Polars DataFrame to pandas DataFrame for the cosine similarity function
	filtered_df = df.filter((std_dev > 0.2).row(0)).to_pandas()

	# Calculate the similarity matrix
	similarity_matrix = cosine_similarity(filtered_df[data.columns].transpose())

	# Convert the similarity matrix back to a Polars DataFrame
	similarity_df = pl.DataFrame(similarity_matrix, columns=data.columns, index=data.columns)

	# Create a graph from the similarity matrix
	adjacency = pd.DataFrame(similarity_matrix > 0.85, columns=data.columns, index=data.columns)

	G = nx.from_pandas_adjacency(adjacency)

	# Remove self-loops from the graph
	self_loops = list(nx.selfloop_edges(G))
	G.remove_edges_from(self_loops)

def plot_volcano(df, fold_change_col='logFC', p_value_col='adj.P.Val', fdr_threshold=0.05, fold_change_threshold=2):
    # Calculate -log10(p-value)
    df['neg_log_pval'] = -np.log10(df[p_value_col])

    # Set color based on thresholds
    df['color'] = 'grey'  # Default color
    df.loc[(df[fold_change_col] > fold_change_threshold) & (df[p_value_col] < fdr_threshold), 'color'] = 'red'
    df.loc[(df[fold_change_col] < -fold_change_threshold) & (df[p_value_col] < fdr_threshold), 'color'] = 'blue'

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df[fold_change_col], df['neg_log_pval'], c=df['color'], alpha=0.5)

    # Add horizontal line for FDR threshold
    plt.axhline(y=-np.log10(fdr_threshold), color='black', linestyle='--')

    # Add vertical lines for fold-change thresholds
    plt.axvline(x=fold_change_threshold, color='black', linestyle='--')
    plt.axvline(x=-fold_change_threshold, color='black', linestyle='--')

    # Labels and title
    plt.xlabel('Log Fold Change')
    plt.ylabel('-Log10(p-value)')
    plt.title('Volcano Plot')

    # Display the plot
    plt.show()

def chdir(data, sampleclass, genes, gamma=1., sort=True, calculate_sig=False, nnull=10, sig_only=False, norm_vector=True):
	"""
	Calculate the characteristic direction for a gene expression dataset
	
	Input:
		data: numpy.array, is the data matrix of gene expression where rows correspond to genes and columns correspond to samples
		sampleclass: list or numpy.array, labels of the samples, it has to be consist of 0, 1 and 2, with 0 being columns to be excluded, 1 being control and 2 being perturbation
				example: sampleclass = [1,1,1,2,2,2]
		genes: list or numpy.array, row labels for genes 
		gamma: float, regulaized term. A parameter that smooths the covariance matrix and reduces potential noise in the dataset
		sort: bool, whether to sort the output by the absolute value of chdir
		calculate_sig: bool, whether to calculate the significance of characteristic directions
		nnull: int, number of null characteristic directions to calculate for significance
		sig_only: bool, whether to return only significant genes; active only when calculate_sig is True
		norm_vector: bool, whether to return a characteristic direction vector normalized to unit vector
	Output:
		A list of tuples sorted by the absolute value in descending order characteristic directions of genes.
			If calculate_sig is set to True, each tuple contains a third element which is the ratio of characteristic directions to null ChDir
	"""
	
	## check input
	data.astype(float)
	sampleclass = np.array(list(map(int, sampleclass)))
	# masks
	m_non0 = sampleclass != 0
	m1 = sampleclass[m_non0] == 1
	m2 = sampleclass[m_non0] == 2

	if type(gamma) not in [float, int]:
		raise ValueError("gamma has to be a numeric number")
	if set(sampleclass) != set([1,2]) and set(sampleclass) != set([0,1,2]):
		raise ValueError("sampleclass has to be a list whose elements are in only 0, 1 or 2")
	# if m1.sum()<2 or m2.sum()<2:
	# 	raise ValueError("Too few samples to calculate characteristic directions")
	if len(genes) != data.shape[0]:
		raise ValueError("Number of genes does not match the demension of the expression matrix")

	## normalize data
	data = data[:, m_non0]
	data = zscore(data) # standardize for each genes across samples

	## start to compute
	n1 = m1.sum() # number of controls
	n2 = m2.sum() # number of experiments

	## the difference between experiment mean vector and control mean vector.
	meanvec = data[:,m2].mean(axis=1) - data[:,m1].mean(axis=1) 

	## initialize the pca object
	pca = PCA(n_components=None)
	pca.fit(data.T)

	## compute the number of PCs to keep
	cumsum = pca.explained_variance_ratio_ # explained variance of each PC
	keepPC = len(cumsum[cumsum > 0.001]) # number of PCs to keep

	v = pca.components_[0:keepPC].T # rotated data 
	r = pca.transform(data.T)[:,0:keepPC] # transformed data

	dd = ( np.dot(r[m1].T,r[m1]) + np.dot(r[m2].T,r[m2]) ) / float(n1+n2-2) # covariance
	sigma = np.mean(np.diag(dd)) # the scalar covariance

	shrunkMats = np.linalg.inv(gamma*dd + sigma*(1-gamma)*np.eye(keepPC))

	b = np.dot(v, np.dot(np.dot(v.T, meanvec), shrunkMats))

	if norm_vector:
		b /= np.linalg.norm(b) # normalize b to unit vector

	grouped = zip([abs(item) for item in b],b,genes)
	if sort:
		grouped = sorted(grouped,key=lambda x: x[0], reverse=True)


	if not calculate_sig: # return sorted b and genes.
		res = [(item[1],item[2]) for item in grouped]
		return res
	else: # generate a null distribution of chdirs
		nu = n1 + n2 - 2
		y1 = np.random.multivariate_normal(np.zeros(keepPC), dd, nnull).T * np.sqrt(nu / chi2.rvs(nu,size=nnull))
		y2 = np.random.multivariate_normal(np.zeros(keepPC), dd, nnull).T * np.sqrt(nu / chi2.rvs(nu,size=nnull))
		y = y2 - y1 ## y is the null of v

		nullchdirs = []
		for col in y.T:
			bn = np.dot(np.dot(np.dot(v,shrunkMats), v.T), np.dot(col,v.T))
			bn /= np.linalg.norm(bn)
			bn = bn ** 2
			bn.sort()
			bn = bn[::-1] ## sort in decending order
			nullchdirs.append(bn)

		nullchdirs = np.array(nullchdirs).T
		nullchdirs = nullchdirs.mean(axis=1)
		b_s = b ** 2 
		b_s.sort()
		b_s = b_s[::-1] # sorted b in decending order
		relerr = b_s / nullchdirs ## relative error
		# ratio_to_null
		ratios = np.cumsum(relerr)/np.sum(relerr)- np.linspace(1./len(meanvec),1,len(meanvec))
		res = [(item[1],item[2], ratio) for item, ratio in zip(grouped, ratios)] 
		print('Number of significant genes: %s'%(np.argmax(ratios)+1))
		if sig_only:
			return res[0:np.argmax(ratios)+1]
		else:
			return res


def paea(chdir, gmtline, case_sensitive=False):
	"""
	Perform principal angle enrichment analysis (PAEA)
	Input:
		chdir, list of tuples: A characteristic direction returned from chdir function
		gmtline: A list of genes from a gene set 
	Output:
		a list of tuples in the format of:
			the principal angle, the p value
	"""
	if len(chdir[0]) == 3: ## output from which calculate_sig is enabled
		chdir = [(item[0],item[1]) for item in chdir]
	if not case_sensitive:
		genes_measured = [gene.upper() for b, gene in chdir]
		gmtline = [gene.upper() for gene in gmtline]
	else: # case sensitive
		genes_measured = [gene for b, gene in chdir]

	if len(set(genes_measured)) != len(chdir):
		raise ValueError('There are duplicated genes in the input genes')
	
	mask = np.in1d(genes_measured, gmtline) # gpos in the R script
	mm = np.where(mask==True)[0]
	m = mask.sum() # number of overlaping genes
	n = len(genes_measured)

	if m > 1 and m < n: # if there is overlap between gene set and genes in chdir
		gsa = np.zeros((n, m)) # Qc in the paper
		for i in range(m):
			gsa[mm[i], i] = 1.

		qb = np.array([b for b,gene in chdir])
		qbqc = np.matrix( np.dot(qb, gsa) ) # Qb.T Qc in paper
		principal_angle = np.linalg.svd(qbqc,compute_uv=False)[0]
		theta = np.arccos(principal_angle)

		# calculation of the null-distribution of principal angles
		m = float(m)
		n = float(n)
		pac = lambda theta: 2.*(1./np.sqrt(2*np.pi))*np.exp((n/2.)*np.log(n/(n - m))+(m/2.)*np.log((n - m)/m)+(1/2.)*np.log(m/(2.*n)*(n - m))+(n-m-1)*np.log(np.sin(theta))+(m-1)*np.log(np.cos(theta)))
		integration_range = np.linspace(0, theta, num=10000, endpoint=True) ## num seems to matter a lot
		p_val = np.trapz(pac(integration_range), integration_range)
		if p_val > 1.:
			p_val = 1.
	else:
		principal_angle = 0.
		p_val = 1.

	return principal_angle, p_val


def paea_wrapper(chdir, gmt_fn, case_sensitive=False, sort=True):
	"""
	A wrapper function for PAEA gene-set enrichment analysis

	Input:
		chdir: characteristic directions computed by chdir function
		gmt_fn: file name of a gene-set library in GMT format
		case_sensitive: whether gene symbols should be considered as case_sensitive
	Output:
		a sorted list of tuples (term, p_val)
	"""
	## check input:
	if not gmt_fn.endswith('.gmt'):
		raise IOError("The gene-set library file is not in GMT format")

	## read gmt into a dict:
	res = []
	with open (gmt_fn) as f:
		for line in f:
			sl = line.strip().split('\t')
			term = sl[0]
			genes = sl[2:]
			if ',' in genes[0]: ## auto detect fuzzy gmts
				genes = [gene.split(',')[0] for gene in genes]
			principal_angle, p_val = paea(chdir, genes, case_sensitive=case_sensitive)
			res.append( (term, p_val) )
	if sort:## sort terms based on p values in ascending order
		res = sorted(res, key=lambda x:x[1])
	else: ## if not sort, return unsorted p_vals only
		res = [p_val for term, p_val in res ]
	return res

