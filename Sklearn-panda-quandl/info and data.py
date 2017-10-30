import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGl')

print(df.head())

#df = df [['Adj. Open','Adj. Low']]
#print(df.head())
#df['HL_PCT'] = (df['Adj. High '] - df['Adj. Close']) / df ['Adj. Close'] *100.0
#df['PCT_change'] = (df['Adj. Close '] - df['Adj. Open']) / df ['Adj. Open'] *100.0

#df = df [['Adj. Close' ,'Adj. Volume']]

#print(df.head())
