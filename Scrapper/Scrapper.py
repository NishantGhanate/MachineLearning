# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:40:42 2018

@author: Nishant
"""

import scrapy

class FirstSpider(scrapy.Spider): 
   name = "first" 
   
   def __init__(self, group = None, *args, **kwargs): 
      super(FirstSpider, self).__init__(*args, **kwargs) 
      self.start_urls = ["http://www.example.com/group/%s" % group]