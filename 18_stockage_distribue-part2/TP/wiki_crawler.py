import scrapy
import happybase
import re
import string

connection = happybase.Connection('localhost')
if 'wiki' in connection.tables():
    connection.delete_table('wiki',True)
connection.create_table('wiki',{'cf':{}})
table_wiki = connection.table('wiki')


class WikiSpider(scrapy.Spider):
    custom_settings = {"DOWNLOAD_HANDLERS": {'s3':None}}
    name = 'wikispider'
    start_urls = ['http://localhost/']
    allowed_domains = ['localhost']

    def parse(self, response):
        url = response.url
        url_absolu = response.urljoin(url)
        if url_absolu.startswith('http://localhost/articles/') and url_absolu.find("%7E")==-1:
            body = string.join(response.xpath('//div[@id="bodyContent"]//*[self::p or self::ul]//text()').extract())
            title=re.sub('.*/(.*)\\.html','\\1',response.url)
            table_wiki.put(title,{'cf:body':body.encode('utf-8')})
        for url in response.xpath("//a/@href").extract():
            url_absolu = response.urljoin(url)
            if url_absolu.startswith('http://localhost/articles/') and url_absolu.find("%7E")==-1:
                yield scrapy.Request(url_absolu)



