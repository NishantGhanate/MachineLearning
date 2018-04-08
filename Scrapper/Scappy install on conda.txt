**** Installing scrapy in anaconda python 3.6+ *******
open anaconda prompt

(base) C:\Users\Nishant> conda install -c scrapinghub scrapy

*******************Starting project ******************************

Select any directory where you want to store your project

(base) C:\Users\Nishant>scrapy startproject sample

 
**********************Lets Start*********************************

open sample folder sample/spiders

copy any of the spider 

open anaconda prompt 

scrapy crawl <spider name> -o file.csv -t csv
scrapy crawl flip -o file.csv -t csv
scrapy crawl flip -o file.json



