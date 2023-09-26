The author of the codes and images contained in this folder is Melpakkam Pradeep.

Folders:
	images	:	Images used in the paper for the coupled reaction PFR for various noise levels.
	cpfr	:	Datasets and codes used to generate and analyze the data for the coupled reaction PFR system. The MATLAB (.m)
			files were used to plot the data.
	pfr	:	Datasets and codes used to generate and analyze the data for the source term PFR system
	heateqn	:	Datasets and codes used to generate and analyze the data for the coupled reaction PFR system. The MATLAB (.m)
			files were used to generate and plot the data.
	
Files:
	my_coupled_pdes	:	IPyNB used to analyze the coupled PFR for various noise levels and hyperparameters.
	heat_eqn_repl	:	IPyNB used to replicate previous results for the Heat Equation for various noise levels and analyze 
				effects of dataset size and some hyperparameters.
	my_pde_series	:	MATLAB file used to generate data in 'cpfr' folder
	my_GKFL0_source	:	IPyNB used to analyze the source term PFR systems.
	my_pde_one	:	MATLAB file used to generate data in 'pfr' folder
	my_GKLF0	:	Modified Python file for GKFL0 to allow for more control over hyperparameters.
	GKLF0		:	Original Python file for GKFL0.