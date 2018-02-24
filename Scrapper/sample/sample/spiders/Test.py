from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector

class MySpider(BaseSpider):
	name = "Nishant" #my spider name 
	allowed_domains = ["craiglist.org"]
	start_urls = ["https://sfbay.craiglist.org/sfc/npo/"]
	
	def parse(self,response):
		hxs = HtmlXPathSelector(response)
		titles = hxs.select("//p")
		for titles in titles:
			title = titles.select("a/text()").extract()
			link = title.select("a/@href").extract()
			print (title , link )
			self.log(title)