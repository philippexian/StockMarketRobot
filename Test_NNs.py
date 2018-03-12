import urllib3
from datetime import datetime

BASE_URL = "https://www.google.com/finance/historical?output=csv&q={0}&startdate=Jan+1%2C+1980&enddate={1}"
symbol_url = BASE_URL.format(
   urllib3.quote('GOOG'), # Replace with any stock you are interested.
   urllib3.quote(datetime.now().strftime("%b+%d,+%Y"), '+')
)

#case when the code is invalid
try:
   f = urllib3.urlopen(symbol_url)
   with open("GOOG.csv", 'w') as fin:
       print (fin, f.read())
except urllib3.HTTPError:
   print ("Fetching Failed: {}".format(symbol_url))