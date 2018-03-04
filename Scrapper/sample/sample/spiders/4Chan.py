from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector

class Chan(BaseSpider):
    name = 'chan'
    start_urls = ['http://www.4chan.org/']

    def parse(self,response):
        for title in response.css('a.boardlink'):
            yield{
                'title' : response.css('a.boardlink::text').extract_first(),
            }