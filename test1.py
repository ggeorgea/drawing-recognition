import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
import ast
import functionMisc
import os
import sys

readdata = []

allData = []
justDescriptors = []
c = 0

for filename in os.listdir('./train_simplified'):
	print(filename," ",c)
	sys.stdout.flush()
	f1 = pd.read_csv("./train_simplified/"+filename)
	d = 0
	for rowind in range(0,f1.values.shape[0]):
		# print(rowind)
		# sys.stdout.flush()
		newformList = []
		filledformList = []
		descriptorList = []

		for stroke in ast.literal_eval(f1['drawing'].values[rowind]):
			currLine = stroke
			newform = []
			for ind in range(0,len(currLine[0])):
				newform.append(np.array([currLine[0][ind],currLine[1][ind]]))
			filledform = functionMisc.fill(newform)
			descriptor = functionMisc.generateDescriptor(filledform,256,256, local = 3)
			# newformList.append(newform)
			# filledformList.append(filledform)
			descriptorList.append(descriptor)

		name = f1['word'].values[rowind]
		#entry = [name,[descriptorList],[newformList],[filledformList]]
		entry2 = [name,descriptorList]
		#break
		#allData.append(entry)
		justDescriptors.append(entry2)
	# 	d=d+1
	# 	if d>200:
	# 		break
	# c=c+1
	# if c>10:
	# 	break
	justDescriptors = np.array(justDescriptors, dtype=object)
	#allData = np.array(allData)
	np.save(("./descriptors/"+filename[:-4]+'_justDescriptors.npy'),justDescriptors)
	#np.save('allData.npy',allData)
	justDescriptors = []











#================================================================================





# df1 = pd.read_csv('angel.csv')
# print()

# currLine = ast.literal_eval(df1['drawing'].values[0])[0]
# newform = []
# for ind in range(0,len(currLine[0])):
# 	newform.append(np.array([currLine[0][ind],currLine[1][ind]]))
# filledform = functionMisc.fill(newform)
# descriptor = functionMisc.generateDescriptor(filledform,256,256, local = 3)


# print(df1.values.shape[0])
# print(df1['word'].values[0])
# for row in df1.values:
# 	print(row.shape)


# print(currLine)
# print("")
# print(filledform)
# print("")
# print(newform)
# print("")
# print(descriptor)

# for entry in df1['drawing'].values:
# 	readdata.append(ast.literal_eval(entry))
# npa = np.array(readdata)
#np.save('allData.npy',allData)



# print(np.fromstring(df1['drawing'].values[0]))