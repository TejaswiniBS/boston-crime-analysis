import numpy as np
import pandas as pd
# from code.results import results
import seaborn as sns
from pyspark.sql.functions import *
from pyspark.sql.types import *
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics

# For WSL2, set the DISPLAY variable(run Mobaxterm)
matplotlib.use('tkagg')

from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results
from utils.one_hot_encoding import one_hot_encoding

def etc(data_dir):
	# ------------------------------------------------------------------------------------------------
	# Predicting UCR_PART
	## ------------------------------------------------------------------------------------------------
	space1_list = ['Lat','Long'] # Accuracy = 0.9890908783869361 F1 score: 0.9890873887741127
	space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = 0.9935081676211407 F1 score: 0.9935112813562653
	time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = 0.9987648657232593 F1 score: 0.9841581390269287
	time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy = 0.9603552383939774 F1 score: 0.9602045812733535
	space1_time_list = space1_list + time1_list # Accuracy = 0.9668209029577904 F1 score: 0.9666683128938756
	space2_time_list = space2_list + time1_list # Accuracy = 0.9476046575516156 F1 score: 0.9471981705156094
	all = space1_list  + space2_list + time1_list # Accuracy = 0.968975066895252  F1 score: 0.9688474378041546

	# # Only Type1 & Type2
	# all = space1_list  + space2_list + time1_list # Accuracy = 0.9652527555738953 F1 score: 0.9651007964872679

	# # # # Only Type2 & Type3
	# all = space1_list  + space2_list + time1_list # Accuracy = 0.972722902637098 F1 score: 0.9726352644418154

	features_list = all + ['OFFENSE_CODE_GROUP']
	prediction_type='UCR_PART'
	#*************************************************************************************************************************************

	# Cleanse Data
	df = cleanse_data(data_dir=data_dir, features_list=features_list, prediction_type=prediction_type)
	df = encoding(df, features_list)


	data=df.toPandas()
	X = data.loc[:, data.columns != 'label']
	Y = data['label']
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.9) # 90% training and 10% test

	model = ExtraTreesClassifier()
	model.fit(x_train, y_train)


	score = model.score(x_train, y_train)
	cv_scores = cross_val_score(model, x_train, y_train, cv=5 )
	ypred = model.predict(x_test)

	# Confusion Matrix
	unique_label = np.unique([y_test, ypred])
	cmtx = pd.DataFrame(
	confusion_matrix(y_test, ypred, labels=unique_label), 
					index=['true:{:}'.format(x) for x in unique_label], 
					columns=['pred:{:}'.format(x) for x in unique_label]
	)
	print(cmtx)

	# print("Score: ", score)
	# print("CV average score: %.2f" % cv_scores.mean())

	print(f'Accuracy: {metrics.accuracy_score(y_test, ypred)}\n')
			# f'F1 score: {metrics.f1_score(y_test, ypred, average = "weighted")}')


	# print(model.feature_importances_)
	#plot graph of feature importances for better visualization
	# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
	# feat_importances.nlargest(10).plot(kind='barh')
	# plt.show()
	# corelation = data.corr()
	# corelation
	# sns.heatmap(corelation)
	# plt.show()

	##########
	# Experiments
	# # ------------------------------------------------------------------------------------------------
	# # Predicting Crime Type
	# ## ------------------------------------------------------------------------------------------------
	# space1_list = ['Lat','Long'] 
	# space2_list = ['DISTRICT','REPORTING_AREA','STREET']
	# time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR']
	# all = space1_list  + space2_list + time1_list # Accuracy = 0.44446894663802444, F1 score: 0.42777359422852407


	# features_list = all
	# prediction_type='OFFENSE_CODE_GROUP'
	# #*************************************************************************************************************************************

	# ------------------------------------------------------------------------------------------------
	# Predicting Severity/Major/Minor
	## ------------------------------------------------------------------------------------------------
	# space1_list = ['Lat','Long'] # Accuracy = 0.4367359598160252, F1 score: 0.4316065681826819
	# space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = 0.4358357625096286, F1 score: 0.42806717269218236
	# time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = 0.3872446091113006, F1 score: 0.37000762954575495
	# time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy =  0.3845858923183867, F1 score: 0.3692104281273839
	# space1_time_list = space1_list + time1_list # Accuracy =  0.42734053497942387, F1 score: 0.4109718238063386
	# space2_time_list = space2_list + time1_list # Accuracy = 0.4293940400851848, F1 score: 0.4125645768575138
	# all = space1_list  + space2_list + time1_list # Accuracy = 0.44446894663802444, F1 score: 0.42777359422852407

	# # Only Type1 & Type2
	# all = space1_list  + space2_list + time1_list # Accuracy = 0.6155674753139095 F1 score: 0.5934144034343128

	# # # Only Type2 & Type3
	# all = space1_list  + space2_list + time1_list # Accuracy =0.5668152860027903 F1 score: 0.5576046416511544

	# # # Only Type1, devided into 7 categories
	# all = space1_list  + space2_list + time1_list # Accuracy = 0.5820707723598255 F1 score: 0.49831688727450973

	# # # Only Type2, devided into 22 categories
	# all = space1_list  + space2_list + time1_list # Accuracy = 0.2630692340302104 F1 score: 0.24627411221021347

	# # # Only Type 1 & 2, devided into 29 categories
	# all = space1_list  + space2_list + time1_list # Accuracy = 0.24947512427702603 F1 score: 0.21809739248855642

	# features_list = all
	# prediction_type='OFFENSE_CODE_GROUP'
	#*************************************************************************************************************************************
