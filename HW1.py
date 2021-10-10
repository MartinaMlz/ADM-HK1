# Say "Hello, World!" With Python

print("Hello, World!")


#Python If-Else

import math
import os
import random
import re
import sys

n = int(input().strip())

if (n%2!=0):
    print("Weird")
elif (n>=2) and (n<=5):
    print("Not Weird")
elif (n>=6) and (n<=20):
    print("Weird")
elif (n>20) and (n<=100):
    print("Not Weird")
    


#Arithmetic Operators

a = int(input())
b = int(input())

s=a+b
d=a-b
p=a*b

print(s)
print(d)
print(p)


#Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    c=a//b
    d=a/b
    
    print(c)
    print(d)


#Loop

if __name__ == '__main__':
    n = int(input())
    
    for i in range(0, n):
        print(i*i)


#Write a function

def is_leap(year):
    leap = False
    
    if(year%4==0):
        if(year%100!=0):
            leap= True 
        elif(year%400==0):
            leap= True
    
    return leap

year = int(input())
print(is_leap(year))


#Print Function

if __name__ == '__main__':
    n = int(input())
    
    l=[]
    for i in range(1, n+1):
        l.append(i)
        
    for i in l:
        print(i, end="")


#List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    perm = [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n]
    print(perm)


#Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))


    m=max(arr)

    arr2=[val for val in arr if val!=m]
    
            
    m2=max(arr2)

    print(m2)


#Nested List
if __name__ == '__main__':
    
    l1=[]
    l2=[]
    n=int(input())
    for _ in range(n):
        name = input()
        l1.append(name)
        score = float(input())
        l2.append(score)
    
    l3= list(dict.fromkeys(l2))
    l3.sort()
    
    min2=l3[1]
    
    l4=[]
    for i in range (0,n):
        if l2[i]==min2:
            l4.append(l1[i])
        
    l4.sort()
    
    for i in l4:
        print(i)
    

#Finding the Percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    
    
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    
    
    query_name = input()
    
    
    i, j, k=student_marks[query_name]
    s=(i+j+k)/3
    s="%0.2f" % (s,)
    
    print(s)


#List

if __name__ == '__main__':
    
    l = []
    n = int(input())
    for i in range(n):
        a = input().split()
        if len(a) == 3:
            eval("l." + a[0] + "(" + a[1] + "," + a[2] + ")")
        elif len(a) == 2:
            eval("l." + a[0] + "(" + a[1] + ")")
        elif a[0] == "print":
            print(l)
        else:
            eval("l." + a[0] + "()")


#Tuples

if __name__ == '__main__':
    n = int(raw_input())
    integer_list = tuple(map(int, raw_input().split()))
    
    h=hash(integer_list)
    
    print(h)


#sWAP cASE

def swap_case(s):
    s2=""
    for i in s:
        if (i.isupper()):
            s2+=i.lower()
        elif (i.islower()):
            s2+=i.upper()
        else:
            s2+=i
    return s2

if __name__ == '__main__':
    s = raw_input()
    result = swap_case(s)
    print result



#String Split and Join

def split_and_join(line):
    a=line.split(" ")
    a="-".join(a)
    
    return a

if __name__ == '__main__':
    line = raw_input()
    result = split_and_join(line)
    print result


#What's Your Name?

def print_full_name(first, last):
    
    print("Hello "+first+" "+last+"! You just delved into python.")

if __name__ == '__main__':
    first_name = raw_input()
    last_name = raw_input()
    print_full_name(first_name, last_name)


#Mutations

def mutate_string(string, position, character):
    l=list(string)
    l[position]=character
    string=''.join(l)
    return string

if __name__ == '__main__':
    s = raw_input()
    i, c = raw_input().split()
    s_new = mutate_string(s, int(i), c)
    print s_new



#Finding a string

def count_substring(string, sub_string):
    count = 0
    for i in range(len(string) - len(sub_string) + 1):
        if string[i:i+len(sub_string)] == sub_string:
            count += 1
    return count

if __name__ == '__main__':
    string = raw_input().strip()
    sub_string = raw_input().strip()
    
    count = count_substring(string, sub_string)
    print count



#String Validator

if __name__ == '__main__':
    s = raw_input()
    
    alnum=False
    alpha=False
    digit=False
    lower=False
    upper=False
    
    
    for i in s:
       if(i.isalnum()): alnum=True
       if(i.isalpha()): alpha=True
       if(i.isdigit()): digit=True
       if(i.islower()): lower=True
       if(i.isupper()): upper=True
       
       
    print(alnum)
    print(alpha)
    print(digit)
    print(lower)
    print(upper)
    


#Text Wrap

import textwrap

def wrap(string, max_width):
    l=textwrap.wrap(string, width=max_width)
    s='\n'.join(l)
    return s

if __name__ == '__main__':
    string, max_width = raw_input(), int(raw_input())
    result = wrap(string, max_width)
    print result



#Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H' #Replace all ______ with rjust, ljust or center.

for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1)) #Top Cone

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)) #Top Pillars

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6)) #Middle Belt

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)) #Bottom Pillars     

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6)) #Bottom Cone



#Designer Door Mat

n, m = map(int, raw_input().split())

path = ".|."
for i in range(1,n,2): 
    print((i * path).center(m, "-"))

print("WELCOME".center(m,"-"))

for i in range(n-2,-1,-2): 
    print((i * path).center(m, "-"))


#String Formatting

def print_formatted(number):
    
    w = len("{0:b}".format(number)) + 1

    for i in range(1, number + 1):
        print "{0:d}".format(i).rjust(w - 1) + "{0:o}".format(i).rjust(w) + "{0:X}".format(i).rjust(w) + "{0:b}".format(i).rjust(w) 

        
    

if __name__ == '__main__':
    n = int(raw_input())
    print_formatted(n)



#Alphabet Rangoli

def print_rangoli(size):
    # your code goes here
    strAlph = 'abcdefghijklmnopqrstuvwxyz'[0:size]
    
    for i in range(size-1, -size, -1):
        print ("--"*abs(i)+ '-'.join(strAlph[size:abs(i):-1] + strAlph[abs(i):size])+"--"*abs(i))

if __name__ == '__main__':
    n = int(raw_input())
    print_rangoli(n)



#Capitalize!

#!/bin/python

import math
import os
import random
import re
import sys

# Complete the solve function below.
def solve(s):
    l=s.split(" ")
    
    complete=""
    for i in l:
       complete+=i.capitalize()+" "
       
    return complete
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = raw_input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()



#The minion game

def minion_game(string):
    vowels = "AEIOU"

    K = S = 0
    for i in range(len(string)):
        if string[i] in vowels:
            K += (len(string)-i)
        else:
            S += (len(string)-i)

    if K > S:
        print("Kevin "+str(K))
    elif K < S:
        print("Stuart "+str(S))
    else:
        print("Draw")

if __name__ == '__main__':
    s = raw_input()
    minion_game(s)



#Merge the Tools!

def merge_the_tools(string, k):
    for x in range(0, len(string), k):
        subs = ""
        for y in string[x : x + k]:
            if y not in subs:
                subs += y          
        print(subs)
    
        
    
if __name__ == '__main__':
    string, k = raw_input(), int(raw_input())
    merge_the_tools(string, k)



#Introduction to Sets

from __future__ import division

def average(array):
    array=set(array)
    a=sum(array)/len(array)
    return a 
    

if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())
    result = average(arr)
    print result



#No Idea!

if __name__ == '__main__':
    n, m = map(int, raw_input().split())
    arr = map(int, raw_input().split())
    A = set(map(int, raw_input().split()))
    B = set(map(int, raw_input().split()))
    
    
    h=0
    for i in arr:
        if i in A:
            h+=1
        elif i in B:
            h-=1
    
    print h



#Symmetric Difference

if __name__ == '__main__':
    M = int(raw_input())
    a = set(map(int, raw_input().split()))
    N = int(raw_input())
    b = set(map(int, raw_input().split()))
    
    inter=a.intersection(b)
    
    union=a.union(b)
    
    diff=union-inter
    
    diff=sorted(diff)
    
    
    for i in diff:
        print(i)



#Set .add()

N = int(input())

countries = set()
for c in range(N):
    s=raw_input()
    countries.add(s)

print(len(countries))



#Set .discard(), .remove() & .pop()

n=int(input())
s = set(map(int, input().split()))
N=int(input())


for i in range (N):
    inp=[]
    inp=list(input().split())
    
    if inp[0]=="pop":
        s.pop()        
    if inp[0]=="remove":    
        s.remove(int(inp[1]))
    if inp[0]=="discard":
        s.discard(int(inp[1]))
    
print (sum(s))


#Set .union() Operation

students_en = int(input())
arrayEN = set(map(int, raw_input().split()))

students_fr = int(input())
arrayFR = set(map(int, raw_input().split()))

u=arrayEN.union(arrayFR)
print(len(u))


#Set .intersection() Operation

students_en = int(input())
arrayEN = set(map(int, raw_input().split()))

students_fr = int(input())
arrayFR = set(map(int, raw_input().split()))

i=arrayEN.intersection(arrayFR)
print(len(i))



#Set .difference() Operation

students_en = int(input())
arrayEN = set(map(int, raw_input().split()))

students_fr = int(input())
arrayFR = set(map(int, raw_input().split()))

u=arrayEN.difference(arrayFR)
print(len(u))



#Set .symmetric_difference() Operation

students_en = int(input())
arrayEN = set(map(int, raw_input().split()))

students_fr = int(input())
arrayFR = set(map(int, raw_input().split()))

u=arrayEN.union(arrayFR)
i=arrayEN.intersection(arrayFR)

print(len(u-i))


#Set Mutations

A = int(input())
setA = set(map(int, input().split()))

N = int(input())

for op in range(N):
    cmd = input().split()
    setB = set(map(int, input().split()))

    if cmd[0] == "intersection_update":
        setA.intersection_update(setB)
    elif cmd[0] == "update":
        setA.update(setB)
    elif cmd[0] == "symmetric_difference_update":
        setA.symmetric_difference_update(setB)
    else:
        setA.difference_update(setB)

print(sum(setA))


#The Captain's Room

S = int(input())
lst_room = map(int, raw_input().split())

captain_room = set()
family_room = set()

for r in lst_room:
    if r in captain_room:
        family_room.add(r)
    else:
        captain_room.add(r)

print(list(captain_room.difference(family_room))[0])



#Check Subset

T = int(input())

for i in range (T):
    nA = int(input())
    A = set(map(int, raw_input().split()))
    nB = int(input())
    B = set(map(int, raw_input().split()))
    
    if A.issubset(B):
        print(True)
    else:
        print(False)



#Check Strict Superset

setStrictA = set(map(int, raw_input().split()))

N = int(input())
flg = 0
for _ in range(N):
    setStrictX = set(map(int, raw_input().split()))

    if setStrictA.intersection(setStrictX) == setStrictX:
        flg += 1
    
print(("True" if flg == N else "False")) 



#collection.Counter()

from collections import Counter

X=input()
size=Counter(list(map(int, raw_input().split())))
N=input()

money=0

for i in range (N):
    req=list(map(int, raw_input().split()))
    
    if size[req[0]]:
        money+=req[1]
        size[req[0]]-=1
        
print(money)



#DefaultDict Tutorial

from collections import defaultdict

n, m=map(int, input().split())

d=defaultdict(list)

for a in range(1, n+1):
    i=input()
    d[i].append(str(a))

    
for b in range(m):
    i=input()
    if d[i]:
        print(" ".join(d[i]))
    else:
        print -1


#Collections.namedtuple()

from collections import namedtuple

ns = int(input())
students = namedtuple('student', input().split())
count = 0
for _ in range(ns):
    c1, c2, c3, c4 = input().split()
    s = students(c1, c2, c3, c4)
    count += int(s.MARKS)

    
m=count/ns
print("{0:.2f}".format(m))



#Collections.OrderedDict()

from collections import OrderedDict

N=int(input())
ord_dic = OrderedDict()

for _ in range(N):
    *item, price = input().split()
    if len(item) > 0:
        str1 = " "
        item = str1.join(item)
        if ord_dic.get(item):
            ord_dic[item] += int(price)
        else:
            ord_dic[item] = int(price)
    else:
        if ord_dic.get(item[0]):
            ord_dic[item[0]] += int(price)
        else:
            ord_dic[item[0]] = int(price)

for item in ord_dic:
    print(item, ord_dic[item])


#Word Order

from collections import defaultdict
d = defaultdict(list)
n=int(input())

for _ in range(n):
    word=input()
    d[word].append(1)

print(len(d.items()))    
    
for i in d.items():
    l=len(i[1])
    print(l, end=" ")



#Collections.deque()

from collections import deque
d = deque()

N=int(input())

for i in range(N):
    op=list(input().split())
    if(op[0]=="pop"):
        d.pop()
    elif(op[0]=="popleft"):
        d.popleft()
    elif (op[0]=="append"):
        d.append(op[1])
    elif (op[0]=="appendleft"):
        d.appendleft(op[1])
        
for n in d:
    print(n, end=" ")



#Pilling Up!

#(I saw the code in terms logic way, then I created the function)
n = int(input())
for _ in range(n): 
    _, cubes = input(), list(map(int, input().split()))
    min_cube=min(cubes)
    min_index = cubes.index(min_cube)
    
    if sorted(cubes[:min_index], reverse=True) == cubes[:min_index] and sorted(cubes[min_index:]) == cubes[min_index:]:
        print("Yes")
    else:
        print("No")



# Calendar Module

import calendar

month, day, year = map(int, input().split())
print((calendar.day_name[calendar.weekday(year, month, day)].upper()))



# Time Delta

from datetime import datetime

def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format)
    t2 = datetime.strptime(t2, time_format)
    return str(int(abs((t1-t2).total_seconds()))) 



#Exception

for _ in range(int(input())):
    try:
        a,b = map(int, input().split()) 
        print(a//b)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)

#Zipped!

students, subjects = map(int, input().split())
subj1 = []
for _ in range(subjects):
    subj1.append(map(float, input().split()))

zipped = zip(*subj1)
for i in zipped:
    print(sum(i)/len(i))


#Athlete sort

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())

    # Athlete Sort in python - Hacker Rank Solution START   
    arr.sort(key = lambda x : x[k])
    for i in arr:
        print(*i,sep=' ')

#ginortS

s = input()
sorted_s_lower = []
sorted_s_upper = []
sorted_s_digit = []
odds = []
even = []

for character in s:
    if character.islower():
        sorted_s_lower.append(character)
    elif character.isupper():
        sorted_s_upper.append(character)
    elif character.isdigit():
        if int(character) % 2 == 0:
            odds.append(character)
        else:
            even.append(character)
        sorted_s_digit = sorted(even) + sorted(odds)

print(''.join((sorted(sorted_s_lower) + sorted(sorted_s_upper) + sorted_s_digit)))


# Detect Floating Point Number

import re

for _ in range(int(input())):
    print(bool(re.match(r'^[+-]?[0-9]*\.[0-9]+$', input())))


# Re.split()

regex_pattern = r"[,.]+"

import re
print("\n".join(re.split(regex_pattern, input())))



# Group(), Groups() & Groupdict()

import re

s = (re.search(r'([a-z0-9])\1+', input()))
if s:
    print(s.group(1))
else:
    print("-1")



# Re.findall() & Re.finditer()

import re

lst = re.findall(r"(?<=[qwrtypsdfghjklzxcvbnm])[aeiouAEIOU]{2,}(?=[qwrtypsdfghjklzxcvbnm])", input())

if len(lst) >= 1:
    for i in lst:
        print(i)
else:
    print("-1")


# Re.start() & Re.end()

#(I saw the code from other solution, it helps me to understand well the other function of the library)
import re 

string = input()
substring = input()

m = re.search(substring, string)
path = re.compile(substring)
if m:
    while m:
        print("({0}, {1})".format(m.start(),m.end()-1))
        m = path.search(string, m.start()+1)
else:
    print("(-1, -1)")


# Regex Substitution

import re

for _ in range(int(input())):
      orType = re.compile(r'(?<= )(\|\|)(?= )')
      andType = re.compile(r'(?<= )(&&)(?= )')
      
      print(orType.sub('or', andType.sub('and', input())))


# Validating Roman Numerals

regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

import re
print(str(bool(re.match(regex_pattern, input()))))



#Validating phone numbers

import re

for _ in range(int(input())):
    if re.match(r'[789]\d{9}$', input()):   
        print('YES') 
    else:  
        print('NO') 


# Validating and Parsing Email Addresses


import re

for _ in range(int(input())):
    x, y = input().split(' ') 
    if re.match(r"<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>", y):
        print(x,y)


# Hex Color Code

import re 

for _ in range(int(input())): 
    rgx = re.findall(r".(#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3})", input()) 
    if rgx:
        for i in rgx:
            print(i)


# HTML Parser - Part 1

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for attr in attrs: 
            print("-> " + attr[0] + " >", attr[1])
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for attr in attrs:
            print("-> " + attr[0] + " >", attr[1])

parser = MyHTMLParser()

for _ in range(int(input())):
    parser.feed(input().strip())


# HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if "\n" in data:
                print('>>> Multi-line Comment')
                print(data)
        else:
                print('>>> Single-line Comment')
                print(data)
  
    def handle_data(self, data):
        if '\n' not in data:
            print(">>> Data")
            print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()



# Validating UID

import re

no_repeats = r"(?!.*(.).*\1)"
two_or_more_upper = r"(.*[A-Z]){2,}"
three_or_more_digits = r"(.*\d){3,}"
ten_alphanumerics = r"[a-zA-Z0-9]{10}"
filters = no_repeats, two_or_more_upper, three_or_more_digits, ten_alphanumerics

for UID in [input() for _ in range(int(input()))]:
    print("Valid" if(all(re.match(fs, UID) for fs in filters)) else "Invalid" )



# Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print("-> " + attr[0], ">", attr[1])

parser = MyHTMLParser()

for i in range(int(input())):
    parser.feed(input())


#XML 1

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    sm = 0
    for children in node.iter():
        sm += len(children.attrib)
    return sm

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


# XML2

import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
    for child in elem:
        depth(child, level + 1) 
        
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

#Standardize Mobile Number using Decorators

def wrapper(f):
    def fun(l):
       f(["+91 " + digits[-10:-5] + " " + digits[-5:] for digits in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 

# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        return map(f, people.sort(key=operator.itemgetter(2)))
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


# Arrays

import numpy

def arrays(arr):
    a = numpy.array(arr, float)
    return a[::-1]

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# Shape and Reshape

import numpy

num_lst = numpy.array(input().split(), int)
print(numpy.reshape(num_lst, (3,3)))


# Transpose and Flatten

import numpy

rows, columns = map(int, input().split())

lst_npy = numpy.array([input().split() for _ in range(rows)], int)
print(lst_npy.transpose())
print(lst_npy.flatten())


# Concatenate

import numpy

n, m, p = map(int, input().split())
lst_num = numpy.array([input().split() for _ in range(n)], int)
lst2_num = numpy.array([input().split() for _ in range(m)], int)

print(numpy.concatenate((lst_num, lst2_num), axis=0))


# Zeros and Ones

import numpy

t = tuple(map(int, input().split()))

print( numpy.zeros((t), dtype = numpy.int) )
print( numpy.ones((t), dtype = numpy.int) )


# Eye and Identity

import numpy

n, m = map(int, input().split())
print(str(numpy.eye(n, m, k=0)).replace('1', ' 1').replace('0', ' 0'))


# Array Mathematics

import numpy

n, m = map(int, input().split())

a = numpy.zeros((n, m), dtype = int)
for i in range(n):
    a[i] = numpy.array([input().split()], int)

b = numpy.zeros((n, m), dtype = int)
for i in range(n):
    b[i] = numpy.array([input().split()], int)

print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(a**b)


# Floor, Ceil and Rint

import numpy

numpy.set_printoptions(sign=' ')

my_array = numpy.array([input().split()], dtype=float)
print(numpy.floor(*my_array))
print(numpy.ceil(*my_array))
print(numpy.rint(*my_array))



# Sum and Prod

import numpy

n, m = map(int, input().split())

a = numpy.zeros((n, m), int)
for i in range(n):
    a[i] = input().split()

print(numpy.prod(numpy.sum(a, axis = 0), axis = None))



# Min and Max

import numpy

n, m = map(int, input().split())

a = numpy.zeros((n, m), int)
for i in range(n):
    a[i] = input().split()

print(numpy.max(numpy.min(a, axis = 1), axis = None))


# Mean, Var, and Std

import numpy

numpy.set_printoptions(legacy='1.13')

n, m = map(int, input().split())

a = numpy.zeros((n, m), int)
for i in range(n):
    a[i] = input().split()

print(numpy.mean(a, axis=1))
print(numpy.var(a, axis=0))
print(numpy.std(a, axis=None))



# Dot and Cross

import numpy

n = int(input())

a = numpy.zeros((n, n), int)
for i in range(n):
    a[i] = input().split()

b = numpy.zeros((n, n), int)
for i in range(n):
    b[i] = input().split()

print(numpy.dot(a, b))



# Inner and Outer

import numpy

a = numpy.array(input().split(), dtype=int)
b = numpy.array(input().split(), dtype=int)

print(numpy.inner(a,b))
print(numpy.outer(a,b))



# Polynomials

import numpy

lst = list(map(float, input().split()))
print(numpy.polyval(lst, int(input())))



# Linear Algebra

import numpy

numpy.set_printoptions(legacy='1.13')

a = numpy.array([input().split() for _ in range(int(input()))], dtype=float)

print(numpy.linalg.det(a))



# Birthday Cake Candles

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    max_candle = max(candles)

    counter = 0
    for c in candles:
        if c == max_candle:
            counter+=1

    return counter

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()



# Number Line Jumps

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    return("YES" if (v1 > v2) and (x2-x1)%(v1-v2) == 0 else "NO")

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()



# Viral Advertising

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    start_people = 5
    cumulative = 0
    for _ in range(n):
        start_people = math.floor(start_people/2)
        cumulative += start_people
        start_people = start_people*3
    return cumulative

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()



# Recursive Digit Sum

import math
import os
import random
import re
import sys

def superDigit(n, k):
    if(len(list(map(int, n))) > 1):
        new_digit = sum(list(map(int, n))) * k
        return superDigit(str(new_digit), 1)
    else:
        return int(n)
  
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


# Insertion Sort - Part 1

import math
import os
import random
import re
import sys

def stamp(arr):
    for elem in arr:
        print(elem, end=" ")
    print("")

def insertionSort1(n, arr):
    for i in range(1, len(arr)):
        x = arr[i]
        j = i-1
        while j >=0 and x < arr[j]:
            arr[j+1] = arr[j]
            stamp(arr)
            j -= 1
        arr[j+1] = x
    stamp(arr)
  
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# Insertion Sort - Part 2

import math
import os
import random
import re
import sys

def stamp(arr):
    for elem in arr:
        print(elem, end=" ")
    print("")

def insertionSort2(n, arr):
    for i in range(1, len(arr)):
        x = arr[i]
        j = i-1
        while j >= 0 and x < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = x
        stamp(arr)
        
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
