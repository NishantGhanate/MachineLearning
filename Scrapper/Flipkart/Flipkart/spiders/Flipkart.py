import scrapy

class Flipkart(scrapy.Spider):
    name = 'flippy'
    start_urls = ['https://www.flipkart.com/mobile-phones-store']

    def parse(self,response):
        for phone in response.css('a.K6IBc-.required-tracking'):
            yield{
                'phone name' : phone.css('div.iUmrbN::text').extract_first(),
                'price' : phone.css('div._3o3r66::text').extract_first(),
            }


