from sklearn import tree
features = [[140,1],[130,1],[150,0] ,[170,0]] #1=smooth 0=bumpy texture of fruit  
labels = [0,0,1,1] #0 = apple 1=orange  labels = output 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features , labels) #
print ( clf.predict([[150,0]]) ) # 

