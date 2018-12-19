"""
CS795
HW3 @author: Yu Zhang
"""

import pandas as pd
import xmltodict
import numpy as np
import xml.etree.ElementTree as et

tree = et.parse('SP500_symbols.xml')
root = tree.getroot()
xml_dict=[]
for child in root:
    xml_dict.append(child.attrib)
#print(xml_dict)
csv_data=pd.read_csv("SP500_ind.csv")
ticker=list(pd.unique(csv_data['Symbol']))
#with open('SP500_symbols.xml') as fd:
#   xml_dict = xmltodict.parse(fd.read(),attr_prefix='')
#print(xml_dict)


def ticker_find(xml_dict, ticker):
    """This function takes in the xml_dict and the list that contains a Symbol (ticker). Return the name of the ticker
    Ex: for ticker “A”, the function returns Agilent Technologies Inc """
    found= False
    for stock in xml_dict:
        if stock['ticker']==ticker:
            found = True
            return stock['name']
    if not found:
        return("No data in SP500")

def calc_avg_open(csv_data, ticker):
    """This function takes in the csv_data and a ticker.
    Return the average opening price for the stock as a float. """
    df=csv_data.loc[lambda x: x.Symbol==ticker, :]
    return(np.mean(df.Open))

def vwap(csv_data, ticker):
    """This function takes in the csv_data and a ticker. Return the volume weighted average price (VWAP) of the stock. In order to do this, first
     find the average price of the stock on each day. Then, multiply that price with the volume on that day. Take the sum of these values. Finally, divide that value by the sum of all the volumes.
    (hint: average price for each day = (high + low + close)/3)
    """
    df=csv_data.loc[lambda x: x.Symbol==ticker, :]
    df.loc[:,'Average_vol']=((df.High+df.Low+df.Close)/3)*df.Volume
    return (np.sum(df['Average_vol'])/np.sum(df.Volume))
    


for name in ticker:
    company_name=ticker_find(xml_dict,name)
    print(company_name,calc_avg_open(csv_data, name),vwap(csv_data, name))

