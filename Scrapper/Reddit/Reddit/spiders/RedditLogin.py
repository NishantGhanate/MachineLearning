from scrapy.spider import BaseSpider
import scrapy

class RedditLogin(BaseSpider):
    name = 'RL'
    start_ulrs = ['https://www.reddit.com/']

    def parse(self,response):
        sitekey = response.css('div.g-recaptcha::attr(data-sitekey)').extract_first()
        data = {
            'data-sitekey': sitekey,
            'user' : 'gibrews',
            'passwd' : 'ssssss',
        }

        yield scrapy.FormRequest(url='https://www.reddit.com/post/login', formdata= data, callback='self.method')

    def method(self,response):
        self.console('OHO')    
