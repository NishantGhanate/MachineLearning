import scrapy
from scrapy.spider import BaseSpider

class Reddit(BaseSpider):

    name = 'Red'
    start_urls = ['https://www.reddit.com/']

    def parse(self,response):
        votes = response.css('div.midcol.unvoted')
        titles = response.css('p.title')
        redditPage = response.css('p.tagline')

        for (vote,title,page) in zip(votes,titles,redditPage):
            yield{
                'votes : ' : vote.css('div.score.likes::text').extract_first(),
                'Title : ' : title.css('a::text').extract_first(),
                'Page : ' : page.css('a.subreddit.hover.may-blank::text').extract_first(),
                'Time : ' : page.css('time::attr(title)').extract_first(),
            }

        nextpage = response.css('span.next-button>a::attr(href)').extract_first()
        if nextpage:
            yield scrapy.Request(url = nextpage , callback = self.parse)    

