import scrapy 

class Flipart(scrapy.Spider):
    name = 'flippy'
    start_urls = ['https://www.flipkart.com/mobile-phones-store']

    def parse(self,response):
         for title in response.css('a.K6IBc-.required-tracking'):
             yield{
                 'product ' : title.css('div.iUmrbN::text').extract_first(),
                 'price' :  title.css('div._3o3r66::text').extract_first(),
             }
