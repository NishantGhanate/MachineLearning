from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
import scrapy

class Reddit(BaseSpider):
    name = 'red'
    start_urls = ['https://www.reddit.com/']

    def parse(self,response):
         Votes  =  response.css('div.midcol.unvoted')
         Titles = response.css('p.title')

         for (vote,titles) in zip( Votes,Titles ):
           yield{
               'vote-count' : vote.css('div.score.unvoted::text').extract_first(),
                'vote-likes' : vote.css('div.score.likes::text').extract_first(),
                'votes-dislike' :  vote.css('div.score.dislikes::text').extract_first(),
                'Title' : titles.css('a::text').extract_first(),
           }


         self.log("Shit")
         nextpage = response.css('span.next-button>a::attr(href)').extract_first() 
         #if nextpage:
          #  yield scrapy.Request(url = nextpage , callback = self.parse)
