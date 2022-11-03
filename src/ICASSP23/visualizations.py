import seaborn as sns
from write_all_results_to_csv import create_all_metrics_df, dataframe_or

logdir = '/share/workhorse3/mshah1/biologically_inspired_models/icassp_logs/'
df = create_all_metrics_df(logdir)

fig1_models = [
    'FMNISTConsistentActivation2048U1L16S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTConsistentActivation2048U1L8S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTConsistentActivation2048U1L4S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTConsistentActivation2048U1L2S0ClsS025ActOptLR02DropoutClassifier',
    'FMNISTConsistentActivation2048U1L1S0ClsS025ActOptLR02DropoutClassifier',
]
keys = ['model'] + [c for c in df.columns if any([k in c for k in ['APGD-0.0','APGD-0.05']])]
fig1_df = df[dataframe_or(df, 'model', fig1_models)][keys]

sns.relplot()
