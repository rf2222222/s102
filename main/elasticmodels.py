#from django.db import models
#from django.contrib.auth.models import User
#from taggit.managers import TaggableManager
#from taggit.models import TaggedItemBase
#from django.utils.safestring import mark_safe
#from main.models import *
from datetime import *
#from django.template.defaultfilters import slugify
#from django.contrib.humanize.templatetags.humanize import *
#from autoslug import AutoSlugField
from dateutil.relativedelta import relativedelta
import dateutil.parser
import pytz
import os
import re
import uuid
import operator


COMPANY_FUNDED_MAX_LENGTH=1000

from pytz import timezone
mytz = pytz.timezone('US/Eastern')

NAME_MAX_LENGTH = 500
COUNTRY_CITY_MAX_LENGTH = 300
PHONE_MAX_LENGTH = 11
GENDER_MAX_LENGTH = 1
MALE = 'M'
FEMALE = 'F'
GENDER_CHOICES = (
    (MALE, 'Male'),
    (FEMALE, 'Female'),
)
UNIVERSITY_MAX_LENGTH = 100
MAJOR_MAX_LENGTH = 300
SPECIALITY_MAX_LENGTH = 200
INTEREST_MAX_LENGTH = 200
INDUSTRY_MAX_LENGTH = 200
FAVORITE_COMPANY_MAX_LENGTH = 200

COMPANY_NAME_MAX_LENGTH = 1000
COMPANY_SHORT_MAX_LENGTH = 500
COMPANY_LONG_MAX_LENGTH = 1000
COMPANY_EMPLOYEE_MAX_LENGTH = 20
COMPANY_CLASS_MAX_LENGTH = 1000
COMPANY_FUNDED_MAX_LENGTH = 300

PRODUCT_NAME_MAX_LENGTH = 500
PRODUCT_CLASS_MAX_LENGTH = 100
PRODUCT_STAGE_MAX_LENGTH = 1000
PRODUCT_MARKET_MAX_LENGTH = 1000

PROGRAM_NAME_MAX_LENGTH = 100

DESCRIPTION_MAX_LENGTH = 200
SHORT_DESCRIPTION_MAX_LENGTH = 2000
LONG_DESCRIPTION_MAX_LENGTH = 10000

from elasticsearch_dsl import Keyword, Mapping, Nested, Text
from datetime import datetime
from elasticsearch_dsl import DocType, Date, Nested, Boolean, \
    analyzer, Completion, Keyword, Text, Integer, Float, MetaField, Object, Long, analysis
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl.query import MultiMatch, Match, Query, SF
from elasticsearch_dsl import Search, Q


myindex='forecast'
connections.configure(
    default={
        'hosts': ['localhost:9201']
    }
)

def get_nonce_id():
    seq=Sequence(_id='nonce_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_resource_continent_id():
    seq=Sequence(_id='resource_continent_id')
    seq.save()
    #pass #print seq._version
    return seq._version + 10000000

def get_syndicate_id():
    seq=Sequence(_id='syndicate_id')
    seq.save()
    #pass #print seq._version
    return seq._version + 10000000

def get_system_id():
    seq=Sequence(_id='system_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_instrument_id():
    seq=Sequence(_id='instrument_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_notification_id():
    seq=Sequence(_id='notification_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_partnership_id():
    seq=Sequence(_id='partnership_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_company_category_list_id():
    seq=Sequence(_id='company_category_list_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_product_category_list_id():
    seq=Sequence(_id='product_category_list_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_category_list_id():
    seq=Sequence(_id='category_list_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_banner_list_id():
    seq=Sequence(_id='banner_list_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_company_investing_event_id():
    seq=Sequence(_id='company_investing_event_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_company_investor_id():
    seq=Sequence(_id='company_investor_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_company_portfolio_id():
    seq=Sequence(_id='company_portfolio_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_market_count_id():
    seq=Sequence(_id='market_count_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_market_dict_id():
    seq=Sequence(_id='market_dict_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_resource_country_status_id():
    seq=Sequence(_id='resource_country_status_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_resource_country_id():
    seq=Sequence(_id='resource_country_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_resource_continent_status_id():
    seq=Sequence(_id='resource_continent_status_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_recent_update_id():
    seq=Sequence(_id='recent_update_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_market_watch_id():
    seq=Sequence(_id='market_watch_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_press_article_id():
    seq=Sequence(_id='press_article_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_article_relation_id():
    seq=Sequence(_id='article_relation_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000


def get_social_feed_id():
    seq=Sequence(_id='social_feed_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10000000

def get_region_id():
    seq=Sequence(_id='region_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10

def get_investor_type_id():
    seq=Sequence(_id='investor_type_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10

def get_job_type_id():
    seq=Sequence(_id='job_type_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10

def get_job_skill_id():
    seq=Sequence(_id='job_skill_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10

def get_job_role_id():
    seq=Sequence(_id='job_role_id')
    seq.save()
    pass #print seq._version
    return seq._version + 50


def get_school_id():
    seq=Sequence(_id='school_id')
    seq.save()
    pass #print seq._version
    return seq._version + 35000

def get_funding_stage_id():
    seq=Sequence(_id='funding_stage_id')
    seq.save()
    pass #print seq._version
    return seq._version + 4000

def get_city_id():
    seq=Sequence(_id='city_id')
    seq.save()
    pass #print seq._version
    return seq._version + 5000

def get_state_id():
    seq=Sequence(_id='state_id')
    seq.save()
    pass #print seq._version
    return seq._version + 1000

def get_country_id():
    seq=Sequence(_id='country_id')
    seq.save()
    pass #print seq._version
    return seq._version + 300

def get_continent_id():
    seq=Sequence(_id='continent_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10

def get_location_id():
    seq=Sequence(_id='location_id')
    seq.save()
    pass #print seq._version
    return seq._version + 10

def get_crawled_article_id():
    seq=Sequence(_id='crawled_article_id')
    seq.save()
    pass #print seq._version
    return seq._version + 8000000

def get_market_id():
    seq=Sequence(_id='market_id')
    seq.save()
    pass #print seq._version
    return seq._version + 120000

def get_resource_id():
    seq=Sequence(_id='resource_id')
    seq.save()
    pass #print seq._version
    return seq._version + 100000000

def get_product_id():
    seq=Sequence(_id='product_id')
    seq.save()
    pass #print seq._version
    return seq._version + 100000000

def get_job_id():
    seq=Sequence(_id='job_id')
    seq.save()
    pass #print seq._version
    return seq._version + 100000000

def get_company_id():
    seq=Sequence(_id='company_id')
    seq.save()
    pass #print seq._version
    return seq._version + 100000000

class Sequence(DocType):
    class Meta:
        index = 'sequence'

class Resource(DocType):
    id=Long()
    owner_id=Integer()
    resource_id=Long()
    company_favorite_count = Integer()
    company_recommendation_count = Integer()
    
    
    resource_type=Text()
    commodity_type=Text()

    is_active = Boolean()
    is_commodity = Boolean()
    is_public = Boolean()
    is_save = Boolean()
    is_confirm = Boolean()

    is_trusted_vc = Boolean()
    is_partner = Boolean()
    is_angel = Boolean()
    
    is_government = Boolean()
    is_tips = Boolean()
    is_rocketpunch = Boolean()
    #is_startup = Boolean()
    is_investor = Boolean()
    
    
    ticker = Text( fields={ 'raw': Keyword() } )
    exchange = Text( fields={ 'raw': Keyword() } )
    sec_cik=Text( fields={ 'raw': Keyword() } )
    sec_cik_int=Text( fields={ 'raw': Keyword() } )
    company_class = Text( fields={ 'raw': Keyword() } )
    partner_order = Integer()



    last_edited_time = Date()
    created_time = Date()
    
    company_name =  Text( fields={ 'raw': Keyword() } )

    investor_class = Text()

    company_short = Text()

    company_long = Text()

    company_industry = Text( fields={ 'raw': Keyword() } )

    score1 = Float()
    score2 = Float()
    score3 = Float()
    score4 = Float()
    score5 = Float()
    
    crawl_source=Text()
    slug = Text( fields={ 'raw': Keyword() } )
    text = Text( fields={ 'raw': Keyword() } )
    
    
    class Meta:
        index = myindex
        
        
    def save(self, *args, ** kwargs):
        if not self.created_time:
            self.created_time = datetime.now()
            self.is_active=True
        if not self.company_class:
            self.company_class='Resource'
        self.last_edited_time = datetime.now()
        if not self.id:
            self.id=get_resource_id()
            self.resource_id=self.id
            self._id='main.resource.' + str(self.id)
            self.my_id=self._id
            self.django_id=str(self.id)
            self.django_ct='main.resource'
            self.text=self.company_name.lower()
            self.is_active=True            
            self.resource_type='Energy'
            self.commodity_type='oil'
        
        super(Resource, self).save(*args, **kwargs)





class Instrument(DocType):
    #resource=models.ForeignKey(Resource,  )
    id=Long()
    resource_id=Long()

    #company=models.ForeignKey(Company,  )
    company_id=Long()

    broker=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    sym=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    text=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    cur=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    exch=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    sec_type=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    trade_freq=Integer()
    mult=Float()
    local_sym=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    
    contract_month=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    expiry=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    ev_rule=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    liquid_hours=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    long_name=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    min_tick=Float()
    time_zone_id=Long()
    trading_hours=Text()
    under_con_id=Long()
    
    created_at = Date()
    updated_at = Date()
    crawl_source=Text()

    
    class Meta:
        index = myindex
    
    def __unicode__(self):
        return '%s' % self.name

    def save(self, *args, **kwargs):
        eastern=timezone('US/Eastern')
       
        if self.created_at == None:
            self.created_at = datetime.now().replace(tzinfo=eastern)  
        self.updated_at = datetime.now().replace(tzinfo=eastern) 
        
        if not self.id:
            self.id=get_instrument_id()
            self._id='main.instrument.' + str(self.id)
            self.my_id=self._id
            self.django_id=str(self.id)
            self.django_ct='main.instrument'
            
        super(Instrument, self).save(*args, **kwargs)



class System(DocType):
    #user = models.ForeignKey(User, primary_key=True)
    version= Text()
    system= Text()
    name=Text()
    c2id=Text()
    c2api=Text()
    c2qty=Integer()
    c2submit=Boolean()
    #c2instrument=models.ForeignKey(Instrument, related_name='c2instrument',  )
    c2instrument_id=Long()

    ibqty=Integer()
    #ibinstrument=models.ForeignKey(Instrument, related_name='ibinstrument',  )
    ibinstrument_id=Long()

    ibsubmit=Boolean()
    trade_freq=Integer()
    ibmult=Integer()
    c2mult=Integer()
    signal=Text()

    class Meta:
        index = myindex
    
    def __unicode__(self):
        return '%s' % self.name

    def save(self, *args, **kwargs):
        eastern=timezone('US/Eastern')
       
        if self.created_at == None:
            self.created_at = datetime.now().replace(tzinfo=eastern)  
        self.updated_at = datetime.now().replace(tzinfo=eastern) 
        
        if not self.id:
            self.id=get_system_id()
            self._id='main.system.' + str(self.id)
            self.my_id=self._id
            self.django_id=str(self.id)
            self.django_ct='main.system'
            #self.text=self.article_title
            #self.slug=str(self.id) + '_' + slugify(self.article_title)
            
        super(System, self).save(*args, **kwargs)



class Feed(DocType):
    #instrument=models.ForeignKey(Instrument)
    id=Long()
    my_id=Text()
    django_ct=Text()
    django_id=Text()
    
    instrument_id=Long()
    
    
    frequency=Integer()
    pct_change=Float()
    settle=Float()
    open_interest=Float()

    date=Date()
    open=Float()
    high=Float()
    low=Float()
    close=Float()
    
    volume=Float()
    wap=Float()
    
    created_at = Date()
    updated_at = Date()
    crawl_source=Text()

    idx_num=Long()
    timestamp=Long()
    
    class Meta:
        index = myindex
        
    def __repr__(self):
        #return '{ "date":"%s", "open":%s, "high":%s, "low":%s, "close":%s,"volume":%s }' % (self.date, self.open, self.high, self.low, self.close, self.volume)
        return '{ "date":"%s",  "close":%s,"volume":%s }' % (self.date,  self.close, self.volume)
        
    def __str__(self):
        return self.__repr__()
    
    def __unicode__(self):
        return self.__repr__()
    
    def save(self, *args, **kwargs):
        eastern=timezone('US/Eastern')
       
        if self.created_at == None:
            self.created_at = datetime.now().replace(tzinfo=eastern)  
        self.updated_at = datetime.now().replace(tzinfo=eastern)  
        try:
            if not self._id:
                mydoc='main.feed.' + str(self.instrument_id) + '|' + str(self.timestamp) + '|' + str(self.frequency)
                self._id=mydoc
        except Exception as e:
            print (e)                
        '''
        if not self.id:
            #self.id=get_feed_id()
            self._id='main.feed.' + str(self.id)
            self.my_id=self._id
            self.django_id=str(self.id)
            self.django_ct='main.feed'
            #self.text=self.article_title
            #self.slug=str(self.id) + '_' + slugify(self.article_title)
            
        '''
         
        super(Feed, self).save(*args, **kwargs)


class Prediction(DocType):
    #instrument=models.ForeignKey(Instrument)
    instrument_id=Long()

    frequency=Integer()
    pred_start_date=Date()
    
    date=Date()
    open=Float()
    high=Float()
    low=Float()
    close=Float()
    volume=Float()
    wap=Float()
    algo_name=Text()
    is_scaled=Boolean()
    created_at = Date()
    updated_at = Date()
    crawl_source=Text()

    class Meta:
        index = myindex
    def __repr__(self):
        #return '{ "date":"%s", "open":%s, "high":%s, "low":%s, "close":%s,"volume":%s }' % (self.date, self.open, self.high, self.low, self.close, self.volume)
        return '{ "date":"%s", "close":%s, "volume":%s }' % (self.date,  self.close, self.volume)
        
    def __str__(self):
        return self.__repr__()
    
    def __unicode__(self):
        return self.__repr__()
    
    def save(self, *args, **kwargs):
        eastern=timezone('US/Eastern')
       
        if self.created_at == None:
            self.created_at = datetime.now().replace(tzinfo=eastern)  
        self.updated_at = datetime.now().replace(tzinfo=eastern)  
        
        '''
        if not self.id:
            #self.id=get_prediction_id()
            self._id='main.prediction.' + str(self.id)
            self.my_id=self._id
            self.django_id=str(self.id)
            self.django_ct='main.prediction'
            #self.text=self.article_title
            #self.slug=str(self.id) + '_' + slugify(self.article_title)
            
        ''' 
        super(Prediction, self).save(*args, **kwargs)


class BidAsk(DocType):
    #instrument=models.ForeignKey(Instrument)
    instrument_id=Long()

    frequency=Integer()
    ask=Float()
    asksize=Float()
    bid=Float()
    bidsize=Float()
    date=Date()
    
    created_at = Date()
    updated_at = Date()
    crawl_source=Text()

    
    class Meta:
        index = myindex
        
    def __unicode__(self):
        return '%s' % self.name

    def save(self, *args, **kwargs):
        eastern=timezone('US/Eastern')
       
        if self.created_at == None:
            self.created_at = datetime.now().replace(tzinfo=eastern)  
        self.updated_at = datetime.now().replace(tzinfo=eastern) 
        super(BidAsk, self).save(*args, **kwargs)

class Signal(DocType):
    session=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    bar_time=Date()
    signal_time=Date()
    order_time=Date()
    execution_time=Date()
    timestamp=Long()

    is_trade_signal=Boolean()
    is_order_sent=Boolean()
    is_exec_done=Boolean()
    frequency=Integer()
    order_type=Text()

    execId=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    orderId=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    order_price=Float()
    qty=Float()    
    exec_price=Float()
    
    sym=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    sym_type=Text()

    
    Open=Float()
    High=Float()
    Low=Float()
    Close=Float()
    Volume=Float()
    
    EMA1_Lookback=Float()
    EMA2_Lookback=Float()
    PricePctChg=Float()
    EMAPctChg=Float()
    ClosePC=Float()
    EMA1=Float()
    EMA1PC=Float()
    EMA2=Float()
    Initial_PSAR=Float()
    PSAR=Float()
    PSARC=Float()
    ATR14=Float()

    ClosePC_grt_PricePctChg=Float()
    EMA1XEMA2=Float()
    EMA1PC_grt_EMAPctChg=Float()
    EMA1_grt_EMA2=Float()
    VP=Float()
    TotalVP=Float()
    TotalVolume=Float()
    VWAP=Float()
    EMA1_grt_VWAP=Float()
    PSART=Float()
    impulse=Float()
    state=Float()
    F=Float()
    _QTY=Float()
    QTY=Float()
    value=Float()
        
    created_at=Date()
    updated_at=Date()
    
    class Meta:
        index = myindex
        
    def save(self, *args, **kwargs):
        eastern=timezone('US/Eastern')
       
        if self.created_at == None:
            self.created_at = datetime.now().replace(tzinfo=eastern)  
        self.updated_at = datetime.now().replace(tzinfo=eastern) 
        super(Signal, self).save(*args, **kwargs)

class Check(DocType):
    bar_time=Date()
    signal_time=Date()
    order_time=Date()
    execution_time=Date()
    timestamp=Long()

    is_trade_signal=Boolean()
    is_order_sent=Boolean()
    is_exec_done=Boolean()
    
    order_type=Text()

    execId=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    orderId=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    order_price=Float()
    qty=Float()    
    exec_price=Float()
    
    sym=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    sym_type=Text()

    
    Open=Float()
    High=Float()
    Low=Float()
    Close=Float()
    Volume=Float()
    
    EMA1_Lookback=Float()
    EMA2_Lookback=Float()
    PricePctChg=Float()
    EMAPctChg=Float()
    ClosePC=Float()
    EMA1=Float()
    EMA1PC=Float()
    EMA2=Float()
    Initial_PSAR=Float()
    PSAR=Float()
    PSARC=Float()

    ClosePC_grt_PricePctChg=Float()
    EMA1XEMA2=Float()
    EMA1PC_grt_EMAPctChg=Float()
    EMA1_grt_EMA2=Float()
    VP=Float()
    TotalVP=Float()
    TotalVolume=Float()
    VWAP=Float()
    EMA1_grt_VWAP=Float()
    PSART=Float()
    impulse=Float()
    state=Float()
    F=Float()
    _QTY=Float()
    QTY=Float()
    value=Float()
        
    
    created_at=Date()
    updated_at=Date()
    
    class Meta:
        index = myindex
        
    def save(self, *args, **kwargs):
        eastern=timezone('US/Eastern')
       
        if self.created_at == None:
            self.created_at = datetime.now().replace(tzinfo=eastern)  
        self.updated_at = datetime.now().replace(tzinfo=eastern) 
        super(Check, self).save(*args, **kwargs)
    
class Execution(DocType):
    execId=Text()
    sym=Text(fields={'raw': Keyword(), 'keyword': Keyword()})
    price=Float()
    qty=Float()
    date=Date()
    timediff=Float()
    realizedPNL=Float()
    commission=Float()
    
    created_at=Date()
    updated_at=Date()
    
    class Meta:
        index = myindex
        
    def save(self, *args, **kwargs):
        eastern=timezone('US/Eastern')
       
        if self.created_at == None:
            self.created_at = datetime.now().replace(tzinfo=eastern)  
        self.updated_at = datetime.now().replace(tzinfo=eastern) 
        super(Execution, self).save(*args, **kwargs)

class Roi(DocType):
    #instrument=models.ForeignKey(Instrument)
    instrument_id=Long()

    frequency=Integer()
    pred_start_date=Date()
    algo_name=Text()
    
    open_date=Date()
    close_date=Date()
    open_price=Float()
    open_qty=Float()
    close_price=Float()
    close_qty=Float()
    direction=Text( )
    pnl=Float()
    pnl_pct=Float()
    is_profitable=Boolean()
    
    is_scaled=Boolean()
    created_at = Date()
    updated_at = Date()
    crawl_source=Text()
    class Meta:
        index = myindex
        
    def __repr__(self):
        #return '{ "date":"%s", "open":%s, "high":%s, "low":%s, "close":%s,"volume":%s }' % (self.date, self.open, self.high, self.low, self.close, self.volume)
        return '{ "open_date":"%s", "close_date":"%s", "is_profitable":%s, "pnl_pct":%s }' % (self.open_date,  self.close_date, self.is_profitable, self.pnl_pct)
        
    def __str__(self):
        return self.__repr__()
    
    def __unicode__(self):
        return self.__repr__()
    
    def save(self, *args, **kwargs):
        eastern=timezone('US/Eastern')
       
        if self.created_at == None:
            self.created_at = datetime.now().replace(tzinfo=eastern)  
        self.updated_at = datetime.now().replace(tzinfo=eastern)  
        super(Roi, self).save(*args, **kwargs)
        
class CurrencyConversion(DocType):
    last_edited_time = Date(
        'date edited',  )
    from_currency_name = Text(fields={'raw': Keyword(),
                                 'keyword': Keyword()})
    to_currency_name = Text(fields={'raw': Keyword(),
                                 'keyword': Keyword()})
    from_currency_price = Float()
    to_currency_price = Float()

    class Meta:
        index = myindex

class Currency(DocType):
    last_edited_time = Date(
        'date edited',  )
    currency_name = Text(fields={'raw': Keyword(),
                                 'keyword': Keyword()})
    currency_price = Float()

    class Meta:
        index = myindex



Feed.init()
Instrument.init()
Prediction.init()
Roi.init()
Resource.init()