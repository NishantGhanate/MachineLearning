from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
import scrapy

class Reddit(BaseSpider):
    name = 'red'
    start_urls = ['https://www.reddit.com/']

    def parse(self,response):
         i = 0 
         nextpage = response.css('span.next-button>a::attr(href)').extract_first()
         for votes in response.css('div.midcol.unvoted'):         
             votes = {
                 'vote-count' : votes.css('div.score.unvoted::text').extract_first(),
                 'vote-likes' : votes.css('div.score.likes::text').extract_first(),
                 'votes-dislike' :  votes.css('div.score.dislikes::text').extract_first(),
             }
             i = i+1
             yield votes
             
         self.log("Shit")
         if nextpage:
            yield scrapy.Request(url = nextpage , callback = self.parse)


        
            

         
                
       
        