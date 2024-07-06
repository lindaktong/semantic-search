import re

re1 = re.compile(f'href="(.*)"')
re2 = re.compile(f'href="(.*?)"')

sample1 = '<a href="google.com">click</a>'
sample2 = '<a href="goo"gle.com">click</a>'

print(re1.findall(sample1))
print(re1.findall(sample2))
print(re2.findall(sample1))
print(re2.findall(sample2))