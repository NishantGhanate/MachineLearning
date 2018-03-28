import scrapy 
import json 

class ElonMusk(scrapy.Spider):
    name = 'Spaceman'
    start_urls = ['https://twitter.com/elonmusk']

    def parse(self,response):
        Tweet =  response.css('div.js-tweet-text-container')
        Reply = response.css('span.ProfileTweet-actionCount')
        for tweet in Tweet:
            yield{
            'Tweets' : tweet.css('p::text').extract_first(),
            

            }
        

	