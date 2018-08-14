""" 
@author [Nishant Ghanate]
@email [nishant7.ng@gmail.com]
@create date 2018-05-30 08:18:23
"""

from sklearn import tree

#[ weight in grams , texture in smooth or pores] 
features = [ [140,1] , [130,1] , [150,0] , [170,0]]

#[0 = orange , 1 = apple ]
lables = [0,0,1,1]

#DecisionTreeClassifier calculates the pattern in data e.g apple tends to be heavier than oraange 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,lables)

print(clf.predict( [ [160,0] ]) )  