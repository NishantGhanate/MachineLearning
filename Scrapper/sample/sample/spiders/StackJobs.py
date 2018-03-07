from scrapy.spider import BaseSpider
import scrapy

class Stack(BaseSpider):

    name = 'Stack'
    allowed_domains = ['stackoverflow.com']
    start_urls =['https://stackoverflow.com/jobs?med=site-ui&ref=jobs-tab']
  

    def parse(self,response):
        jobs = response.css('h2.g-col10')
        offers = response.css('div.-perks.g-row')
        companyNames = response.css('div.-company.g-row')
        i +=1
        next_page = response.css('div.pagination>a::attr(href)')[i].extract()

        for (job,offer,company) in zip (jobs,offers,companyNames):
            yield{
                'Job ' : job.css('a::text').extract_first(),
                'offers ':offer.css('span.-salary::text').extract_first(),
                'Compnay ':company.css('div.-name::text').extract_first(), 
            }

        self.log(next_page)
        if next_page:
            next_page = response.urljoin(next_page)
            self.log(next_page)  
            yield scrapy.Request(url = next_page , callback = self.parse)


        
            
