import scrapy 
import json 

class ElonMusk(scrapy.Spider):
    name = 'Spaceman'
    scroll_link = 'https://twitter.com/i/profiles/show/elonmusk/timeline/tweets?include_available_features=1&include_entities=1&max_position=960982935175245824&reset_error_state=false'
    start_urls = [scroll_link]

    def parse(self,response):
        pageload = json.loads(response.text) 
        for data in response.css('p.TweetTextSize.TweetTextSize--normal.js-tweet-text.tweet-text'):
            yield{
                'Tweets ' :  data.css('p.TweetTextSize.TweetTextSize--normal.js-tweet-text.tweet-text::text').extract_first(),             
            }
        if  pageload["has_more_items"]:
             yield scrapy.Request(url=self.scroll_link,callback=self.parse) 
	