__author__ = 'claire'
import re
from bs4 import BeautifulSoup
from urllib2 import urlopen

rest = []
page = 'http://www.just-eat.dk/area/2300-koebenhavn-s'
page = urlopen(page)
soup = BeautifulSoup(page.read(), 'html.parser')
res = soup.findAll('a', {'class': 'extLink'})
for i in res:
    rest.append({'name': i.attrs['href']})

for r in rest:
    for tag in res:
        if r['name'] == tag.attrs['href']:
            try:
                r['count'] = int(tag.attrs['title'].split()[0])
            except:
                pass

for r in rest:
    for tag in res:
        if r['name'] == tag.attrs['href']:
            try:
                r['rating'] = re.findall('rating-\d+', str(tag.span))[0]
            except:
                pass

# remove duplicates
nodupes = set(frozenset(d.items()) for d in rest)
nodupes = [dict(s) for s in nodupes]

# only include those with count key
counts = [n for n in nodupes if 'count' in n]
print 'Entries (that have been reviewed):', len(counts)

# sort by key
newlist = sorted(counts, key=lambda k: k['count'], reverse=True)
for i in newlist:
    print 'Number of reviews: %d\tRating: %s\tLink:%s' % (
        i['count'], i['rating'], 'http://www.just-eat.dk/' + i['name'] + '/menu')