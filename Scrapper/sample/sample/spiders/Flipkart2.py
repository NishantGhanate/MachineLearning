from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector

class Flipkart2(BaseSpider):
    name = 'Flip'
    allowed_domains = ['https://www.flipkart.com']
    start_urls = ['https://www.flipkart.com/mobile-phones-store']
   

    def parse(self,response):
        domain = 'https://www.flipkart.com'
        hxs = HtmlXPathSelector(response)
        pages = response.css('a._2AkmmA._1eFTEo::attr(href)')
        if pages :
            for page in pages:                
                self.log(  page)
               # yield scrapy.Request(url=page,callback=parse_details)
        
"""
    def parse_details(self,response):
        self.log('Now scraping + response.url')
        for phoneInfo in response.css('a.K6IBc-.required-tracking'):
             yield{
                 'product ' : phoneInfo.css('div.iUmrbN::text').extract_first(),
                 'price' :  phoneInfo.css('div._3o3r66::text').extract_first(),
             }"""