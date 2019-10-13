# ===== PROBLEM1 =====
# Exercise 1 - Introduction - Say "Hello, World!" With Python
print("Hello, World!")


# Exercise 2 - Introduction - Python If-Else
def isOdd(n):
    if n%2!=0:
        print('Weird')
    else:
            if 2<=n and n<=5:
                print('Not Weird')
            else:
                if 6<=n and n<=20:
                    print('Weird')
                else:
                    if 20<n:
                        print('Not Weird')
isOdd(n)


# Exercise 3 - Introduction - Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
def Funzione(a,b):
    print((a+b))
    print((a-b))
    print((a*b))
Funzione(a,b)


# Exercise 4 - Introduction - Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
def Divi(a,b):
    print(int(a/b))
    print(a/b)
Divi(a,b)


# Exercise 5 - Introduction - Loops
if __name__ == '__main__':
    n = int(input())
for i in range(n):
    print(i*i)
    i+=1

    
# Exercise 6 - Introduction - Write a function
def is_leap(year):
    leap = False
    
    return(year%400==0 or year%4==0 and year%100!=0)
year = int(input())
print(is_leap(year))


# Exercise 7 - Introduction - Print Function
if __name__ == '__main__':
    n = int(input())
l=[]
for i in range(1,n+1):
    l.append(str(i))
print("".join(l))



# Exercise 8 - Basic data types - List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
print ([[a,b,c] for a in range(0,x+1) for b in range(0,y+1) for c in range(0,z+1) if a + b + c != n ])


# Exercise 9 - Basic data types - Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    lista = list(map(int, input().split()))
print(sorted(list(set(lista)))[-2])


# Exercise 10 - Basic data types - Nested Lists
if __name__ == '__main__':
    marksheet = []
    score=[]
    for i in range(int(input())):
        marksheet.append([input(), float(input())])
        score.append(marksheet[i][1])
    second_low=sorted(set(score))[1]
    names=[]
    for i in range(len(marksheet)):
        if marksheet[i][1]==second_low:
            names.append(marksheet[i][0])
    
    names.sort()
    for i in names:
        print(i)
        
        
# Exercise 11 - Basic data types - Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    print(format(sum(student_marks.get(query_name))/len(scores),'.2f'))
    
    
# Exercise 12 - Basic data types - Lists
if __name__ == '__main__':
    N = int(input())
    a=[]
    for i in range(N):
        lista=(input().split())
        if lista[0]=='insert':
            a.insert(int(lista[1]),int(lista[2]))
        if lista[0]=='print':
            print(a)
        if lista[0]=='remove':
            a.remove(int(lista[1]))
        if lista[0]=='append':
            a.append(int(lista[1]))
        if lista[0]=='sort':
            a.sort()
        if lista[0]=='pop':
            a.pop()
        if lista[0]=='reverse':
            a.reverse()
            
            
# Exercise 13 - Basic data types - Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = tuple(map(int, input().split()))
    print(hash(integer_list))
    
    
# Exercise 14 - Strings - sWAP cASE
def swap_case(s):
    return(s.swapcase())


# Exercise 15 - Strings - String Split and Join
def split_and_join(line):
    return("-".join(line.split()))


# Exercise 16 - Strings - What's Your Name?
def print_full_name(a, b):
    print("Hello "+a+' '+b+"! You just delved into python.")

    
# Exercise 17 - Strings - Mutations
def mutate_string(string, position, character):
    fine = string[:(position)] + character + string[position+1:]
    return(fine)


# Exercise 18 - Strings - Find a string
def count_substring(string, sub_string):
    return(sum([1 for i in range(0, len(string) - len(sub_string) + 1) if (string[i:(len(sub_string)+i)] == sub_string)]))


# Exercise 19 - Strings - String Validators
if __name__ == '__main__':
    s = input()
    print(any(i.isalnum() for i in s))
    print(any(i.isalpha() for i in s))
    print(any(i.isdigit() for i in s))
    print(any(i.islower() for i in s))
    print(any(i.isupper() for i in s))


# Exercise 20 - Strings - Text Alignment

# Exercise 21 - Strings - Text Wrap
def wrap(string, max_width):
    a=textwrap.wrap(string,max_width)
    print(*a, sep = "\n")
    return ''


# Exercise 22 - Strings - Designer Door Mat
n, m = map(int,input().split())
rombo = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
print('\n'.join(rombo + ['WELCOME'.center(m, '-')] + rombo[::-1]))


# Exercise 23 - Strings - String Formatting
# Exercise 24 - Strings - Alphabet Rangoli
# Exercise 25 - Strings - Capitalize!
# Exercise 26 - Strings - The Minion Game
def minion_game(string):
    vocali=['A','E','I','O','U']
    Kevin=0
    Stuart=0
    for i in range(len(string)):
        if vocali.count(s[i])!=0:
            Kevin += len(string)-i
        else:
            Stuart += len(string)-i
    if Kevin > Stuart:
        print('Kevin',Kevin)
    elif Kevin == Stuart:
        print("Draw")
    elif Kevin<Stuart:
        print('Stuart',Stuart)


# Exercise 27 - Strings - Merge the Tools!
import textwrap
def merge_the_tools(string, k):
        s1= textwrap.wrap(string, width=k)
        for i in s1:
            print("".join(list(dict.fromkeys(i))))
    


# Exercise 28 - Sets - Introduction to Sets
def average(array):
    return(sum(set(arr))/len(set(arr)))


# Exercise 29 - Sets - No Idea!
n, m = map(int,input().split())
s = map(int, input().split())
A = set(map(int, input().split()))
B = set(map(int, input().split()))
print(sum([(i in A) - (i in B) for i in s]))



# Exercise 30 - Sets - Symmetric Difference
M= int(input())
A=list(map(int, input().split()))
A1=set(A)
N=int(input())
B=list(map(int, input().split()))
B1=set(B)

sd=sorted(list(A1.symmetric_difference(B1)))
print(*sd, sep = "\n") 


# Exercise 31 - Sets - Set .add()
n=int(input())
try: 
    my_list = [] 
      
    while True: 
        my_list.append(input())
except: 
    print(len(list(set(my_list))))

# if input is not-integer, just print the list 


# Exercise 32 - Sets - Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for i in range(N):
    lista=(input().split())
    if lista[0]=='pop':
        s.pop()
    if lista[0]=='discard':
            s.discard(int(lista[1]))
    if lista[0]=='remove':
            s.remove(int(lista[1]))
print(sum(s))


# Exercise 33 - Sets - Set .union() Operation
N=int(input())
s = set(map(int, input().split()))
M=int(input())
t = set(map(int, input().split()))
print(len(s.union(t)))


# Exercise 34 - Sets - Set .intersection() Operation
N=int(input())
s = set(map(int, input().split()))
M=int(input())
t = set(map(int, input().split()))
print(len(s.intersection(t)))

# Exercise 35 - Sets - Set .difference() Operation
N=int(input())
s = set(map(int, input().split()))
M=int(input())
t = set(map(int, input().split()))
print(len(s.difference(t)))


# Exercise 36 - Sets - Set .symmetric_difference() Operation
N=int(input())
s = set(map(int, input().split()))
M=int(input())
t = set(map(int, input().split()))
print(len(s.symmetric_difference(t)))


# Exercise 37 - Sets - Set Mutations
# Exercise 38 - Sets - The Captain's Room
K= int(input())
lista = sorted(list(map(int,input().strip().split())))
a=[]
for i in range(len(lista)):
    if lista[i]==lista[i-1]:
        a.append(lista[i])
a=set(a)
lista=set(lista)
r=lista.difference(a)
r=list(r)
print(r[0])


# Exercise 39 - Sets - Check Subset
for _ in range(int(input())):
    x, a, z, b = input(), set(input().split()), input(), set(input().split())
    print(a.issubset(b))

    
# Exercise 40 - Sets - Check Strict Superset
A=set(map(int, input().split()))
n=int(input())
B=set(map(int, input().split()))
C=set(map(int, input().split()))
print(len(C.difference(A))==0 & len(B.difference(A))==0)


# Exercise 41 - Collections - collections.Counter()
from collections import Counter #it's like a disctionary 
n=int(input())
lista= Counter(list(map(int,input().split())))
numero = int(input())
income = 0 
#the income variable will be increased if the customer
#buys a pair of shoes
for i in range(numero):
    size, price = map(int, input().split())
    if lista[size]: 
        income = income + price
        lista[size] -= 1

print(income)


# Exercise 42 - Collections - DefaultDict Tutorial
from collections import defaultdict

n, m = map(int, input().split())
d = defaultdict(lambda: -1)

for i in range(1, n+1): 
    word = input()
    d[word] = d[word] + ' ' + str(i) if word in d else str(i)

for _ in range(m):
    print(d[input()])

    
# Exercise 43 - Collections - Collections.namedtuple()
# Exercise 44 - Collections - Collections.OrderedDict()
# Exercise 45 - Collections - Word Order
# Exercise 46 - Collections - Collections.deque()
# Exercise 47 - Collections - Company Logo
# Exercise 48 - Collections - Piling Up!


# Exercise 49 - Date time - Calendar Module
import calendar
m,d,y=map(int,input().split())
print(list(calendar.day_name)[calendar.weekday(y, m, d)].upper())
#for this exercise i whatched the solution


# Exercise 50 - Date time - Time Delta

# Exercise 51 - Exceptions -
n=int(input())
for i in range(n):
    
    try:
        a,b = map(int,input().split())
    except ValueError as e:
        print('Error Code:',e)
        continue
    try:
        print(a//b)
    except ZeroDivisionError as e:
        print('Error Code:',e)



# Exercise 52 - Built-ins - Zipped!
import itertools
m,n=map(int,input().split())
X=[]
for i in range(n):
    X.append(list(map(float,input().strip().split())))

for x in zip(*X):
    print(format(sum(list(x))/n, '.1f'))

    
# Exercise 53 - Built-ins - Athlete Sort
# Exercise 54 - Built-ins - Ginorts


# Exercise 55 - Map and lambda function
cube = lambda x:x**3
def fibonacci(n):
    if n==0:
        return []
    if n==1:
        return [0]
    lis=[0,1]
    for i in range(2,n):
        lis.append((lis[i-2]+lis[i-1]))
    return lis
#this piece of code produce a list with the cube 
#of the firsts n numbers of Fibonacci's succession

#Regex
#for this kind of exercises
#I searched more or less everything online
# Exercise 56 - Regex - Detect Floating Point Number
import re
n=int(input())
for i in range(n):
    s=input()
    print(bool(re.match(r'[+-]?\d*[.]\d+$',s)))



# Exercise 57 - Regex - Re.split()
regex_pattern = r"[.,]"	


# Exercise 58 - Regex - Group(), Groups() & Groupdict()
# Exercise 59 - Regex - Re.findall() & Re.finditer()
import re
v = "aeiou"
c = "qwrtypsdfghjklzxcvbnm"
m = re.findall(r"(?<=[%s])([%s]{2,})[%s]" % (c, v, c), input(), flags = re.I)
print('\n'.join(m or ['-1']))


# Exercise 60 - Regex - Re.start() & Re.end()
# Exercise 61 - Regex - Regex Substitution
# Exercise 62 - Regex - Validating Roman Numerals
# Exercise 63 - Regex - Validating phone numbers
# Exercise 64 - Regex - Validating and Parsing Email Addresses
# Exercise 65 - Regex - Hex Color Code
# Exercise 66 - Regex - HTML Parser - Part 1
# Exercise 67 - Regex - HTML Parser - Part 2
# Exercise 68 - Regex - Detect HTML Tags, Attributes and Attribute Values
# Exercise 69 - Regex - Validating UID
# Exercise 70 - Regex - Validating Credit Card Numbers
# Exercise 71 - Regex - Validating Postal Codes
# Exercise 72 - Regex - Matrix Script
# Exercise 73 - Xml - XML 1 - Find the Score
# Exercise 74 - Xml - XML 2 - Find the Maximum Depth
# Exercise 75 - Closures and decorators - Standardize Mobile Number Using Decorators
# Exercise 76 - Closures and decorators - Decorators 2 - Name Directory


# Exercise 77 - Numpy - Arrays
def arrays(arr):
    arr.reverse()
    b = numpy.array(arr,float)
    return b
    
    
# Exercise 78 - Numpy - Shape and Reshape
import numpy
arr = list(map(int, input().split()))
my_array = numpy.array(arr)
print(numpy.reshape(my_array,(3,3)))


# Exercise 79 - Numpy - Transpose and Flatten
import numpy

n, m = map(int, input().split())
array = numpy.array([input().strip().split() for _ in range(n)], int)
print (array.transpose())
print (array.flatten())


# Exercise 80 - Numpy - Concatenate
import numpy as np
a, b, c = map(int,input().split())
arrA = np.array([input().split() for _ in range(a)],int)
arrB = np.array([input().split() for _ in range(b)],int)
print(np.concatenate((arrA, arrB), axis = 0))



# Exercise 81 - Numpy - Zeros and Ones
import numpy
mat = tuple(map(int, input().split()))
print (numpy.zeros(mat, dtype = numpy.int))
print (numpy.ones(mat, dtype = numpy.int))


# Exercise 82 - Numpy - Eye and Identity
import numpy
print(str(numpy.eye(*map(int,input().split()))).replace('1',' 1').replace('0',' 0'))


# Exercise 83 - Numpy - Array Mathematics
import numpy as np
n, m = map(int, input().split())
a, b = (np.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')



# Exercise 84 - Numpy - Floor, Ceil and Rint
import numpy
numpy.set_printoptions(sign=' ')

a = numpy.array(input().split(),float)

print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))



# Exercise 85 - Numpy - Sum and Prod
import numpy
N, M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)],int)
print(numpy.prod(numpy.sum(A, axis=0), axis=0))


# Exercise 86 - Numpy - Min and Max
import numpy
N, M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)],int)
print(numpy.max(numpy.min(A, axis=1), axis=0))


# Exercise 87 - Numpy - Mean, Var, and Std
# Exercise 88 - Numpy - Dot and Cross
import numpy
a=int(input())
arr1=numpy.array([list(map(int,input().split())) for _ in range(a)])
arr2=numpy.array([list(map(int,input().split())) for _ in range(a)])
print(numpy.dot(arr1,arr2))


# Exercise 89 - Numpy - Inner and Outer
import numpy as np
A = np.array(input().split(), int)
B = np.array(input().split(), int)
print(np.inner(A,B), np.outer(A,B), sep='\n')



# Exercise 90 - Numpy - Polynomials
import numpy
n = list(map(float,input().split()))
m = input()
print(numpy.polyval(n,int(m)))



# Exercise 91 - Numpy - Linear Algebra
import numpy
n=int(input())
a=numpy.array([input().split() for _ in range(n)],float)
numpy.set_printoptions(legacy='1.13')
print(numpy.linalg.det(a))



# ===== PROBLEM2 =====

# Exercise 92 - Challenges - Birthday Cake Candles
def birthdayCakeCandles(ar):
    return(ar.count(max(ar)))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    ar_count = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(ar)
    fptr.write(str(result) + '\n')
    fptr.close()

    

# Exercise 93 - Challenges - Kangaroo
#I tried this exercise many many times
#I spoke with other guys about this exercies and I know 
#that they solved it in another way 

#My solution fail in 3 tests because I had to set 
#a range that if is bigger involves an error
#At the same time I think that could be a good solution 
s = list(input().split())
x1=int(s[0])
v1=int(s[1])
x2=int(s[2])
v2=int(s[3])
seq1=list(range(x1,3000000,v1))
seq2=list(range(x2,3000000,v2))
a='NO'
b=len(seq1)
c=len(seq2)
d=min(b,c)
s1d=seq1[:d]
s2d=seq2[:d]
diff=[s1d - s2d for s1d, s2d in zip(s1d, s2d)]
if x1!=x2 and v1==v2:
    a='NO'
else:
    for i in range(d):
       if diff[i]==0:
        a='YES'
        break
print(a)

# Exercise 94 - Challenges - Viral Advertising
n=int(input())
pop=2
tot = []
for i in range(n-1):
    add=pop*3//2
    pop=add
    tot.append(pop)
print(sum(tot)+2) 
#I add 2 at the print because at the start we have 2 person
#the loop calculates how many people are reached by the advertising



# Exercise 95 - Challenges - Recursive Digit Sum
nk = input().split()
n = nk[0]
k = int(nk[1])
if k==100000:
    k=1
#this if reduce the value of k because if k is a multiple
#of 10 the sum of the digits doesn't change
#i tried to make a for-loop to reduce the value of key 
#but there were code's problem 
somma=(n*k)
tot1=sum(list(map(int, list(n))))*k
for i in range(1,1000):
    somma=list(map(int, list(somma)))
    tot1=sum(somma)
    somma=str(tot1)
    if tot1<10:
        print(tot1)
        break
        
        
# Exercise 96 - Challenges - Insertion Sort - Part 1
# Exercise 97 - Challenges - Insertion Sort - Part 2
n = int(input())
arr = list(input().rstrip().split())
for i in range(1,n):
    if int(arr[i])<int(arr[i-1]):
        a=arr[i]
        arr.pop(i)
        for j in range(0,n):
            if int(a)<int(arr[j]):
                arr.insert(j,a)
                break
    print(' '.join(arr))  
#in the first loop the code check if the list is sorted 
#if the list is not sorted, the code takes the value that gives problem 
#puts it into a variable and removes it
#in the second loop the code searches the right position to insert the value







