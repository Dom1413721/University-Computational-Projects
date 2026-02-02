import math
import random
import pandas as pd
import xlsxwriter
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
#import openpyxl

def cdf(x):
    #cdf of N(0,1) variable.
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def cut(deck=list(range(1,53)), cutdist="bin", mean=26, stdev=0):
    #cuts a deck of cards with a given distribution
    dist=[0]*(n+1)
    if cutdist == "bin":
        p=mean/n
        for i in range(n+1):
            dist[i] = (p**i)*((1-p)**(n-i))*(math.comb(n,i))
    elif cutdist == "norm":
        if stdev != 0:
            for i in range(1,n+2):
                dist[i-1] = cdf((i+0.5-mean)/stdev)-cdf((i-0.5-mean)/stdev) #dist list is used as a weight function, not probability function, so doesn't need to sum to 1
        else:
            dist[mean-1] = 1
    elif cutdist == "unif":
        dist = [1]*(n+1)

    n=len(deck)
    cut=random.choices(list(range(n+1)), weights = dist)[0]
    pack1 = deck[:cut]
    pack2 = deck[cut:]
    newdeck = pack2 + pack1
   
    #print("here is the new order: ", newdeck)
    return(newdeck)

def randomriffle(deck=list(range(1,53)),cutdist="bin",packetdist="prop", mean=26, stdev=0):
    #riffle shuffles a deck of cards, taking both the distribution for the start cut and the dropping of packets as inputs.
    #bottom card is card number 1
    n = len(deck)

    dist=[0]*(n+1)
    if cutdist == "bin":
        p=mean/n
        for i in range(n+1):
            dist[i] = (p**i)*((1-p)**(n-i))*(math.comb(n,i))
    elif cutdist == "norm":
        if stdev != 0:
            for i in range(1,n+2):
                dist[i-1] = cdf((i+0.5-mean)/stdev)-cdf((i-0.5-mean)/stdev)
        else:
            dist[mean-1] = 1
    elif cutdist == "unif":
        dist = [1]*(n+1)
        
    cut=random.choices(list(range(n+1)), weights = dist)[0]
    
    pack1 = deck[:cut]
    pack2 = deck[cut:]    

    newdeck = []

    if packetdist == "prop":
        for i in range(n):
            p1 = len(pack1)/(len(pack1)+len(pack2))

            if random.random() <= p1:
                newdeck.append(pack1[0])
                pack1 = pack1[1:]
            else:
                newdeck.append(pack2[0])
                pack2 = pack2[1:]
                
    elif packetdist == "equal":
        for i in range(n):
            if len(pack1) == 0:
                newdeck.append(pack2[0])
                pack2=pack2[2:]
            elif len(pack2) == 0:
                newdeck.append(pack1[0])
                pack1=pack1[1:]                 
            elif random.random() <= 0.5:
                newdeck.append(pack1[0])
                pack1=pack1[1:]
            else:
                newdeck.append(pack2[0])
                pack2=pack2[1:]
            
    #print("here is the new order: ", newdeck)
    return(newdeck)

def randomrifflecut(deck=list(range(1,53)), cutdist1="bin", mean1=26, stdev1=0, packetdist="prop", cutdist2="bin", mean2=26, stdev2=0):
    #performs a riffle shuffle, then cuts the deck

    newdeck = randomriffle(deck, cutdist1, packetdist, mean1, stdev1)

    newdeck = cut(newdeck, cutdist2, mean2, stdev2) 

    #print("here is the new order: ", newdeck)
    return(newdeck)

def test(sample1, sample2,alpha,direction):
    #performs a t test on 2 samples
    #always it fails when p<0.05
    n1 = len(sample1) #size of samples
    n2 = len(sample2)
    m1 = sum(sample1)/n1 #mean of samples
    m2 = sum(sample2)/n2
    s1 = math.sqrt((sum([(i-m1)**2 for i in sample1])/(n1-1))) #standard deviation of samples
    s2 = math.sqrt((sum([(i-m2)**2 for i in sample2])/(n2-1)))
    stde = math.sqrt((s1**2)/n1+(s2**2)/n2)
    tstat = (m1-m2)/stde
    DoF = (stde**4)/((s1**4)/((n1**2)*(n1-1))+(s2**4)/((n2**2)*(n2-1)))#

    if direction == ">":
        p = 1 - t.cdf(tstat, DoF)
        cutoff = t.ppf(1-alpha,DoF)
        
    elif direction == "<":
        p=t.cdf(tstat, DoF)
        cutoff = t.ppf(alpha,DoF)

    return([n1,n2,m1,m2,s1,s2,tstat,cutoff,p])

def disorder(deck):
    #finds different measures of disorder of a deck of cards

    n=len(deck)

    #Entropy
    count1 =[0 for i in range(n-1)]
    count2 = [0 for i in range(math.floor(n/2))]

    for i in range(n):
        x1= (deck[(i+1) % n]-deck[i]) % n
        x2=min(abs(x1),abs(n-x1))
        count1[x1-1]+=1
        count2[x2-1]+=1

    count1 = [i/n for i in count1]
    count2 = [i/n for i in count2]

    
    for i in range(n-1):
        if count1[i] > 0:
            count1[i] = -count1[i]*math.log(count1[i],2)

    for i in range(math.floor(n/2)):
        if count2[i] > 0:
            count2[i] = -count2[i]*math.log(count2[i],2)

    U1 = sum(count1)
    U2 = sum(count2)
    
    #print("Information entropy (positive distance):", U1)
    #print("Information entropy (absolute distance):", U2)    

    #No. of transpositions
    transpos=0
    deck2=list(deck)
    for i in range(1,n+1):
        if deck2[i-1] != i:
            deck2[deck2.index(i)]=deck2[i-1]
            deck2[i-1]=i
            transpos+=1
    #print("Minimum number of transpositions needed to get back to perfectly ordered:", transpos)

    #No of disjoint cycles
    l1cycles = 0
    for i in range(n):
        if deck[i] == i+1:
            l1cycles+=1

    c = n - transpos - l1cycles
    #print("Number of disjoint cycles:", c)


    #No of transpos to get back to original cycle
    start=list(range(1,n+1))
    ctranspos=transpos
    for j in range(2,n+1):
        transpos2=0
        deck2=list(deck)
        start2 = list(start)
        start2[:n-j] = list(start[j-1:])
        start2[n-j+1:] = list(start[:j-1])
        for i in range(1,n+1):
            if deck2[i-1] != start2[i-1]:
                deck2[deck2.index(start2[i-1])]=deck2[i-1]
                deck2[i-1]=start2[i-1]
                transpos2+=1
        if transpos2 < ctranspos:
            ctranspos=transpos2

    #print("Minimum number of transpositions needed to get back to perfectly ordered up to cycling the deck:", ctranspos)

    #Spearman's rank
    deck2=list(deck)
    for i in range(n):
        deck2[i] = (deck[i]-i-1)**2

    r = 1 - (6*sum(deck2))/(n*(n**2-1))

    #print("Spearman's rank correlation coefficient:", r)

    #rising sequences
    seq=0
    pos=n+1
    for i in range(1,n+1):
        if deck.index(i) < pos:
            seq+=1
        pos=deck.index(i)


    #print("Number of rising sequences:", seq)
        
    return([U1,U2,transpos,c, ctranspos, r, seq])

def getresults(n,k,m):
    #function to return the measures of disorder for a sample of m decks of n cards after k shuffles.
    #it's hardcoded to use binomial distribution for the cut and proportional distribution for the riffle (fits experimental data the best)

    ans=[]
    deck = list(range(1,n+1))
    for i in range(m):
        for j in range(k):
            deck = randomriffle(deck, "bin", "prop", n/2, 1)
        ans.append(disorder(deck))

    return(ans)
            
def trans(matrix):
    #transposes a matrix
    transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    return transposed

def doresults():
    #simulates shuffling 1000 decks of 52 cards 7 times, then finds the average of all the measures of disorder.
    results = [[[0 for i in range(7)] for j in range(1000)] for k in range(10)]

    for i in range(10):
        results[i] = getresults(52,i+1,1000)

    sums = [[0 for i in range(7)] for k in range(10)]
    
    for i in range(10):
        for j in range(7):
            for k in range(1000):
                sums[i][j] = float(sums[i][j])+results[i][k][j]

    for i in range(10):
        for j in range(7):
            sums[i][j]=sums[i][j]/1000

    return(sums)

def normweights(mean,stdev,n):
    #returns weights (unnormalised probabilities) of the closest integer to a normal variable being the integers 26 away from its mean
    weights=list(range(53))

    for i in range(53):
        weights[i] = cdf((i+0.5-mean)/stdev)-cdf((i-0.5-mean)/stdev)

    return weights


#uses above functions to get some samples, and exports the data to a spreadsheet
sample=[]
riffle1=[]
riffle2=[]
riffle3=[]
riffle4=[]
riffle5=[]
riffle6=[]
riffle7=[]
riffle8=[]
riffle9=[]
riffle10=[]

ans=[[0 for a in range(10)] for b in range(10)]
for n in range (10,101,10):
    print(n)
    for i in range(1000):
        sample.append(disorder(random.sample(range(1,n+1),n))[0])
        deck1 = list(range(1,n+1))
        deck2 = list(range(1,n+1))
        deck3 = list(range(1,n+1))
        deck4 = list(range(1,n+1))
        deck5 = list(range(1,n+1))
        deck6 = list(range(1,n+1))
        deck7 = list(range(1,n+1))
        deck8 = list(range(1,n+1))
        deck9 = list(range(1,n+1))
        deck10 = list(range(1,n+1))
        
        for k in range(1):
            deck1 = randomriffle(deck = deck1)
        for k in range(2):
            deck2 = randomriffle(deck = deck2)
        for k in range(3):
            deck3 = randomriffle(deck = deck3)
        for k in range(4):
            deck4 = randomriffle(deck = deck4)
        for k in range(5):
            deck5 = randomriffle(deck = deck5)
        for k in range(6):
            deck6 = randomriffle(deck = deck6)
        for k in range(7):
            deck7 = randomriffle(deck = deck7)
        for k in range(8):
            deck8 = randomriffle(deck = deck8)
        for k in range(9):
            deck9 = randomriffle(deck = deck9)
        for k in range(10):
            deck10 = randomriffle(deck = deck10)
        
        riffle1.append(disorder(deck1)[0])
        riffle2.append(disorder(deck2)[0])
        riffle3.append(disorder(deck3)[0])
        riffle4.append(disorder(deck4)[0])
        riffle5.append(disorder(deck5)[0])
        riffle6.append(disorder(deck6)[0])
        riffle7.append(disorder(deck7)[0])
        riffle8.append(disorder(deck8)[0])
        riffle9.append(disorder(deck9)[0])
        riffle10.append(disorder(deck10)[0])

    if n==50:
        x=[test(sample,riffle1,0.05,">"),
        test(sample,riffle2,0.05,">"),
        test(sample,riffle3,0.05,">"),
        test(sample,riffle4,0.05,">"),
        test(sample,riffle5,0.05,">"),
        test(sample,riffle6,0.05,">"),
        test(sample,riffle7,0.05,">"),
        test(sample,riffle8,0.05,">"),
        test(sample,riffle9,0.05,">"),
        test(sample,riffle10,0.05,">")]
        
    ans[0][int(n/10-1)] = test(sample,riffle1,0.05,">")[8]
    ans[1][int(n/10-1)] = test(sample,riffle2,0.05,">")[8]
    ans[2][int(n/10-1)] = test(sample,riffle3,0.05,">")[8]
    ans[3][int(n/10-1)] = test(sample,riffle4,0.05,">")[8]
    ans[4][int(n/10-1)] = test(sample,riffle5,0.05,">")[8]
    ans[5][int(n/10-1)] = test(sample,riffle6,0.05,">")[8]
    ans[6][int(n/10-1)] = test(sample,riffle7,0.05,">")[8]
    ans[7][int(n/10-1)] = test(sample,riffle8,0.05,">")[8]
    ans[8][int(n/10-1)] = test(sample,riffle9,0.05,">")[8]
    ans[9][int(n/10-1)] = test(sample,riffle10,0.05,">")[8]
    
print(x)
print(ans)  

ansdf=pd.DataFrame(ans)

writer = pd.ExcelWriter('presdiffn2.xlsx', engine='xlsxwriter')
ansdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()


#earlier testing and exporting to spreadsheets
"""x=[test(sample,riffle1,0.05,">"),
test(sample,riffle2,0.05,">"),
test(sample,riffle3,0.05,">"),
test(sample,riffle4,0.05,">"),
test(sample,riffle5,0.05,">"),
test(sample,riffle6,0.05,">"),
test(sample,riffle7,0.05,">"),
test(sample,riffle8,0.05,">"),
test(sample,riffle9,0.05,">"),
test(sample,riffle10,0.05,">")]

xdf=pd.DataFrame(x)





x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
x9 = []
x10 = []
x=0

for k in range(1,2):
    for mu in range(21,32):
        x=0
        for i in range(10000):
            deck = list(range(1,53))
            for j in range(k):
                deck = randomriffle(deck=deck, cutdist = 'norm', packetdist = 'prop', mean = mu, stdev = math.sqrt(13))
            x+=disorder(deck)[0]
        x=x/1000
        if k == 1:
            x1.append(x)
        elif k == 2:
            x2.append(x)
        elif k == 3:
            x3.append(x)
        elif k == 4:
            x4.append(x)
        elif k == 5:
            x5.append(x)
        elif k == 6:
            x6.append(x)
        elif k ==7:
            x7.append(x)
        elif k == 8:
            x8.append(x)
        elif k == 9:
            x9.append(x)
        elif k == 10:
            x10.append(x)"""


"""print(x1.index(max(x1)),
      x2.index(max(x2)),
      x3.index(max(x3)),
      x4.index(max(x4)),
      x5.index(max(x5)),
      x6.index(max(x6)),
      x7.index(max(x7)),
      x8.index(max(x8)),
      x9.index(max(x9)),
      x10.index(max(x10)))


x=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
xdf = pd.DataFrame(x)
writer = pd.ExcelWriter('muinitial.xlsx', engine='xlsxwriter')
xdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()"""

"""x1=[]

for mu in range(20,33):
    x=0
    print(mu)
    for i in range(10000):
        deck = list(range(1,53))
        deck = randomrifflecut(deck,cutdist = 'norm', packetdist = 'prop', mean = mu, stdev = 5)
        x+=disorder(deck)[0]
    x=x/10000
    x1.append(x)
        
print(x1)

mu = [0,4,8,12,16,20,24,28,32,36,40,44,48,52]


x1cut=[]

for k in range(1,2):
    for mu in range(0,52,4):
        x=0
        for i in range(10000):
            deck = list(range(1,53))
            deck = randomrifflecut(deck,cutdist = 'norm', packetdist = 'prop', mean = mu, stdev = math.sqrt(13))
            x+=disorder(deck)[0]
        x=x/10000
        x1cut.append(x)


x1=[]
for sigma in range(0,9):
    print(sigma)
    x=0
    for i in range(10000):
        deck = list(range(1,53))
        deck = randomriffle(deck,cutdist = 'norm', packetdist = 'prop', mean = 26, stdev = sigma)
        x+=disorder(deck)[0]
    x=x/10000
    x1.append(x)


        
print(x1)

sigma = [0,1,2,3,4,5,6,7,8]
#plt.plot(mu,x1)
#hold on
#plt.plot(mu,x2)
#plt.show()
"""

"""
sample=[]
riffle1=[]
riffle2=[]
riffle3=[]
riffle4=[]
riffle5=[]
riffle6=[]
riffle7=[]
riffle8=[]
riffle9=[]
riffle10=[]
            
        
            


for i in range(1000):
    sample.append(disorder(random.sample(range(1,53),52))[0])
    deck1 = list(range(1,53))
    deck2 = list(range(1,53))
    deck3 = list(range(1,53))
    deck4 = list(range(1,53))
    deck5 = list(range(1,53))
    deck6 = list(range(1,53))
    deck7 = list(range(1,53))
    deck8 = list(range(1,53))
    deck9 = list(range(1,53))
    deck10 = list(range(1,53))
        
    for i in range(1):
        deck1 = randomrifflecut(deck = deck1,cutdist="norm",mean=26,stdev=5)
    for i in range(2):
        deck2 = randomrifflecut(deck = deck2,cutdist="norm",mean=26,stdev=5)
    for i in range(3):
        deck3 = randomrifflecut(deck = deck3,cutdist="norm",mean=26,stdev=5)
    for i in range(4):
        deck4 = randomrifflecut(deck = deck4,cutdist="norm",mean=26,stdev=5)
    for i in range(5):
        deck5 = randomrifflecut(deck = deck5,cutdist="norm",mean=26,stdev=5)
    for i in range(6):
        deck6 = randomrifflecut(deck = deck6,cutdist="norm",mean=26,stdev=5)
    for i in range(7):
        deck7 = randomrifflecut(deck = deck7,cutdist="norm",mean=26,stdev=5)
    for i in range(8):
        deck8 = randomrifflecut(deck = deck8,cutdist="norm",mean=26,stdev=5)
    for i in range(9):
        deck9 = randomrifflecut(deck = deck9,cutdist="norm",mean=26,stdev=5)
    for i in range(10):
        deck10 = randomrifflecut(deck = deck10,cutdist="norm",mean=26,stdev=5)
        
    riffle1.append(disorder(deck1)[0])
    riffle2.append(disorder(deck2)[0])
    riffle3.append(disorder(deck3)[0])
    riffle4.append(disorder(deck4)[0])
    riffle5.append(disorder(deck5)[0])
    riffle6.append(disorder(deck6)[0])
    riffle7.append(disorder(deck7)[0])
    riffle8.append(disorder(deck8)[0])
    riffle9.append(disorder(deck9)[0])
    riffle10.append(disorder(deck10)[0])
    
#sample = pd.DataFrame(sample)
#sns.kdeplot(data=sample)


#sampledf = pd.DataFrame(sample)
#sns.kdeplot(data=sampledf)


x=[test(sample,riffle1,0.05,">"),
test(sample,riffle2,0.05,">"),
test(sample,riffle3,0.05,">"),
test(sample,riffle4,0.05,">"),
test(sample,riffle5,0.05,">"),
test(sample,riffle6,0.05,">"),
test(sample,riffle7,0.05,">"),
test(sample,riffle8,0.05,">"),
test(sample,riffle9,0.05,">"),
test(sample,riffle10,0.05,">")]

xdf=pd.DataFrame(x)

writer = pd.ExcelWriter('normtest.xlsx', engine='xlsxwriter')
xdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()


avg0=[0,0,0,0,0,0,0,0,0,0]
avg1=[0,0,0,0,0,0,0,0,0,0]
avg2=[0,0,0,0,0,0,0,0,0,0]
avg3=[0,0,0,0,0,0,0,0,0,0]
avg4=[0,0,0,0,0,0,0,0,0,0]
avg5=[0,0,0,0,0,0,0,0,0,0]
avg6=[0,0,0,0,0,0,0,0,0,0]
for j in range(1,16):
    for i in range(1000):
        deck=range(1,53)
        for k in range(j):
            deck = randomriffle(deck,"bin","prop",26,0)
        x=disorder(deck)
        avg0[j-1]+=x[0]/1000
        avg1[j-1]+=x[1]/1000
        avg2[j-1]+=x[2]/1000
        avg3[j-1]+=x[3]/1000
        avg4[j-1]+=x[4]/1000
        avg5[j-1]+=x[5]/1000
        avg6[j-1]+=x[6]/1000


for i in range(10000):
    deck=random.sample(range(1,53),52)
    x=disorder(deck)
    for j in range(7):
        rand[j]+=x[j]/10000




                   


norm=[]
mean = 
for i in range(1,53):
        norm[i-1] = cdf((i+0.5-mean)/stdev)-cdf((i-0.5-mean)/stdev)"""




"""rand1000=[0,0,0,0,0,0,0]
rand5000=[0,0,0,0,0,0,0]
rand10000=[0,0,0,0,0,0,0]
rand50000=[0,0,0,0,0,0,0]


for i in range(1000):
    deck=random.sample(range(1,17),16)
    x=disorder(deck)
    x[5]=abs(x[5])
    for j in range(7):
        rand1000[j]+=x[j]
        rand5000[j]+=x[j]
        rand10000[j]+=x[j]
        rand50000[j]+=x[j]

for i in range(4000):
    deck=random.sample(range(1,17),16)
    x=disorder(deck)
    x[5]=abs(x[5])
    for j in range(7):
        rand5000[j]+=x[j]
        rand10000[j]+=x[j]
        rand50000[j]+=x[j]

for i in range(5000):
    deck=random.sample(range(1,17),16)
    x=disorder(deck)
    x[5]=abs(x[5])
    for j in range(7):
        rand10000[j]+=x[j]
        rand50000[j]+=x[j]

for i in range(40000):
    deck=random.sample(range(1,17),16)
    x=disorder(deck)
    x[5]=abs(x[5])
    for j in range(7):
        rand50000[j]+=x[j]

rand1000=[i/1000 for i in rand1000]
rand5000=[i/5000 for i in rand5000]
rand10000=[i/10000 for i in rand10000]
rand50000=[i/50000 for i in rand50000]

randoms = [rand1000, rand5000, rand10000, rand50000]                 
        
df1000 = pd.DataFrame(rand1000)
df5000 = pd.DataFrame(rand5000)
df10000 = pd.DataFrame(rand10000)
df50000 = pd.DataFrame(rand50000)

df=pd.DataFrame(randoms)

writer = pd.ExcelWriter('randomsamples2.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()"""

"""U1vals=[]
U2vals=[]

for i in range(1000):
    deck=random.sample(range(1,53),52)
    x=disorder(deck)
    U1vals.append(x[0])
    U2vals.append(x[1])



U1vsU2 = [U1vals, U2vals]
U1vsU2df = pd.DataFrame(U1vsU2)

writer = pd.ExcelWriter('U1vsU2df.xlsx', engine='xlsxwriter')
U1vsU2df.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()"""

"""tvals=[]
ctvals=[]

for i in range(1000):
    deck=random.sample(range(1,53),52)
    x=disorder(deck)
    tvals.append(x[2])
    ctvals.append(x[4])


tvsct = [tvals,ctvals]
tvsctdf = pd.DataFrame(tvsct)

writer = pd.ExcelWriter('tvsct.xlsx', engine='xlsxwriter')
tvsctdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()"""


"""U1vals=[]
rvals=[]

for i in range(1000):
    deck=random.sample(range(1,53),52)
    x=disorder(deck)
    U1vals.append(x[0])
    rvals.append(x[5])


rvsU1 = [rvals,U1vals]
rvsU1df = pd.DataFrame(rvsU1)

writer = pd.ExcelWriter('rvsU1.xlsx', engine='xlsxwriter')
rvsU1df.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()


Dweights = normweights(26.056, 2.974601995,52)
Cweights = normweights(26.27230047, 3.188591467,52)
Gweights = normweights(25.756, 4.07071824,52)


weights = [Dweights, Cweights, Gweights]
weightsdf = pd.DataFrame(weights)

writer = pd.ExcelWriter('weights.xlsx', engine='xlsxwriter')
weightsdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()"""

"""rand=[0,0,0,0,0,0,0]

for i in range(50000):
    deck=random.sample(range(1,53),52)
    x=disorder(deck)
    for j in range(7):
        rand[j]+=x[j]


rand = [i/50000 for i in rand]"""

"""
avg0=[0,0,0,0,0,0,0,0,0,0]
avg1=[0,0,0,0,0,0,0,0,0,0]
avg2=[0,0,0,0,0,0,0,0,0,0]
avg3=[0,0,0,0,0,0,0,0,0,0]
avg4=[0,0,0,0,0,0,0,0,0,0]
avg5=[0,0,0,0,0,0,0,0,0,0]
avg6=[0,0,0,0,0,0,0,0,0,0]
for j in range(1,16):
    for i in range(50000):
        deck=range(1,53)
        for k in range(j):
            deck = randomriffle(deck,"bin","prop",26,0)
        x=disorder(deck)
        avg0[j-1]+=x[0]
        avg1[j-1]+=x[1]
        avg2[j-1]+=x[2]
        avg3[j-1]+=x[3]
        avg4[j-1]+=x[4]
        avg5[j-1]+=x[5]
        avg6[j-1]+=x[6]

        
avg0=[i/50000 for i in avg0]
avg1=[i/50000 for i in avg1]
avg2=[i/50000 for i in avg2]
avg3=[i/50000 for i in avg3]
avg4=[i/50000 for i in avg4]
avg5=[i/50000 for i in avg5]
avg6=[i/50000 for i in avg6]
"""        

"""sample=[]
riffle1=[]
riffle2=[]
riffle3=[]
riffle4=[]
riffle5=[]
riffle6=[]
riffle7=[]
riffle8=[]
riffle9=[]
riffle10=[]


for i in range(1000):
    sample.append(disorder(random.sample(range(1,53),52))[0])
    deck1 = list(range(1,53))
    deck2 = list(range(1,53))
    deck3 = list(range(1,53))
    deck4 = list(range(1,53))
    deck5 = list(range(1,53))
    deck6 = list(range(1,53))
    deck7 = list(range(1,53))
    deck8 = list(range(1,53))
    deck9 = list(range(1,53))
    deck10 = list(range(1,53))
        
    for i in range(1):
        deck1 = randomriffle(deck = deck1)
    for i in range(2):
        deck2 = randomriffle(deck = deck2)
    for i in range(3):
        deck3 = randomriffle(deck = deck3)
    for i in range(4):
        deck4 = randomriffle(deck = deck4)
    for i in range(5):
        deck5 = randomriffle(deck = deck5)
    for i in range(6):
        deck6 = randomriffle(deck = deck6)
    for i in range(7):
        deck7 = randomriffle(deck = deck7)
    for i in range(8):
        deck8 = randomriffle(deck = deck8)
    for i in range(9):
        deck9 = randomriffle(deck = deck9)
    for i in range(10):
        deck10 = randomriffle(deck = deck10)
        
    riffle1.append(disorder(deck1)[0])
    riffle2.append(disorder(deck2)[0])
    riffle3.append(disorder(deck3)[0])
    riffle4.append(disorder(deck4)[0])
    riffle5.append(disorder(deck5)[0])
    riffle6.append(disorder(deck6)[0])
    riffle7.append(disorder(deck7)[0])
    riffle8.append(disorder(deck8)[0])
    riffle9.append(disorder(deck9)[0])
    riffle10.append(disorder(deck10)[0])
    
#sample = pd.DataFrame(sample)
#sns.kdeplot(data=sample)


sampledf = pd.DataFrame(sample)
sns.kdeplot(data=sampledf)


x=[test(sample,riffle1,0.05,">"),
test(sample,riffle2,0.05,">"),
test(sample,riffle3,0.05,">"),
test(sample,riffle4,0.05,">"),
test(sample,riffle5,0.05,">"),
test(sample,riffle6,0.05,">"),
test(sample,riffle7,0.05,">"),
test(sample,riffle8,0.05,">"),
test(sample,riffle9,0.05,">"),
test(sample,riffle10,0.05,">")]

xdf=pd.DataFrame(x)

writer = pd.ExcelWriter('bintestU1pres.xlsx', engine='xlsxwriter')
xdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()

"""
"""
sample=[]
riffle1=[]
riffle2=[]
riffle3=[]
riffle4=[]
riffle5=[]
riffle6=[]
riffle7=[]
riffle8=[]
riffle9=[]
riffle10=[]


for i in range(1000):
    sample.append(disorder(random.sample(range(1,53),52))[5])
    deck1 = list(range(1,53))
    deck2 = list(range(1,53))
    deck3 = list(range(1,53))
    deck4 = list(range(1,53))
    deck5 = list(range(1,53))
    deck6 = list(range(1,53))
    deck7 = list(range(1,53))
    deck8 = list(range(1,53))
    deck9 = list(range(1,53))
    deck10 = list(range(1,53))
        
    for i in range(1):
        deck1 = randomriffle(deck = deck1)
    for i in range(2):
        deck2 = randomriffle(deck = deck2)
    for i in range(3):
        deck3 = randomriffle(deck = deck3)
    for i in range(4):
        deck4 = randomriffle(deck = deck4)
    for i in range(5):
        deck5 = randomriffle(deck = deck5)
    for i in range(6):
        deck6 = randomriffle(deck = deck6)
    for i in range(7):
        deck7 = randomriffle(deck = deck7)
    for i in range(8):
        deck8 = randomriffle(deck = deck8)
    for i in range(9):
        deck9 = randomriffle(deck = deck9)
    for i in range(10):
        deck10 = randomriffle(deck = deck10)
        
    riffle1.append(disorder(deck1)[5])
    riffle2.append(disorder(deck2)[5])
    riffle3.append(disorder(deck3)[5])
    riffle4.append(disorder(deck4)[5])
    riffle5.append(disorder(deck5)[5])
    riffle6.append(disorder(deck6)[5])
    riffle7.append(disorder(deck7)[5])
    riffle8.append(disorder(deck8)[5])
    riffle9.append(disorder(deck9)[5])
    riffle10.append(disorder(deck10)[5])
    
#sample = pd.DataFrame(sample)
#sns.kdeplot(data=sample)

#sampledf = pd.DataFrame(sample)
#sns.kdeplot(data=sampledf)


x=[test(sample,riffle1,0.05,"<"),
test(sample,riffle2,0.05,"<"),
test(sample,riffle3,0.05,"<"),
test(sample,riffle4,0.05,"<"),
test(sample,riffle5,0.05,"<"),
test(sample,riffle6,0.05,"<"),
test(sample,riffle7,0.05,"<"),
test(sample,riffle8,0.05,"<"),
test(sample,riffle9,0.05,"<"),
test(sample,riffle10,0.05,"<")]

xdf=pd.DataFrame(x)

writer = pd.ExcelWriter('bintestr.xlsx', engine='xlsxwriter')
xdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()"""

"""sample=[]
riffle1=[]
riffle2=[]
riffle3=[]
riffle4=[]
riffle5=[]
riffle6=[]
riffle7=[]
riffle8=[]
riffle9=[]
riffle10=[]
riffle11=[]
riffle12=[]
riffle13=[]
riffle14=[]
riffle15=[]


for i in range(1000):
    sample.append(disorder(random.sample(range(1,53),52))[6])
    deck1 = list(range(1,53))
    deck2 = list(range(1,53))
    deck3 = list(range(1,53))
    deck4 = list(range(1,53))
    deck5 = list(range(1,53))
    deck6 = list(range(1,53))
    deck7 = list(range(1,53))
    deck8 = list(range(1,53))
    deck9 = list(range(1,53))
    deck10 = list(range(1,53))
    deck11 = list(range(1,53))
    deck12 = list(range(1,53))
    deck13 = list(range(1,53))
    deck14 = list(range(1,53))
    deck15 = list(range(1,53))
        
    for i in range(1):
        deck1 = randomriffle(deck = deck1)
    for i in range(2):
        deck2 = randomriffle(deck = deck2)
    for i in range(3):
        deck3 = randomriffle(deck = deck3)
    for i in range(4):
        deck4 = randomriffle(deck = deck4)
    for i in range(5):
        deck5 = randomriffle(deck = deck5)
    for i in range(6):
        deck6 = randomriffle(deck = deck6)
    for i in range(7):
        deck7 = randomriffle(deck = deck7)
    for i in range(8):
        deck8 = randomriffle(deck = deck8)
    for i in range(9):
        deck9 = randomriffle(deck = deck9)
    for i in range(10):
        deck10 = randomriffle(deck = deck10)
    for i in range(11):
        deck11 = randomriffle(deck = deck11)
    for i in range(12):
        deck12 = randomriffle(deck = deck12)
    for i in range(13):
        deck13 = randomriffle(deck = deck13)
    for i in range(14):
        deck14 = randomriffle(deck = deck14)
    for i in range(15):
        deck15 = randomriffle(deck = deck15)
        
    riffle1.append(disorder(deck1)[6])
    riffle2.append(disorder(deck2)[6])
    riffle3.append(disorder(deck3)[6])
    riffle4.append(disorder(deck4)[6])
    riffle5.append(disorder(deck5)[6])
    riffle6.append(disorder(deck6)[6])
    riffle7.append(disorder(deck7)[6])
    riffle8.append(disorder(deck8)[6])
    riffle9.append(disorder(deck9)[6])
    riffle10.append(disorder(deck10)[6])
    riffle11.append(disorder(deck11)[6])
    riffle12.append(disorder(deck12)[6])
    riffle13.append(disorder(deck13)[6])
    riffle14.append(disorder(deck14)[6])
    riffle15.append(disorder(deck15)[6])
    
#sample = pd.DataFrame(sample)
#sns.kdeplot(data=sample)

#sampledf = pd.DataFrame(sample)
#sns.kdeplot(data=sampledf)


x=[test(sample,riffle1,0.05,">"),
test(sample,riffle2,0.05,">"),
test(sample,riffle3,0.05,">"),
test(sample,riffle4,0.05,">"),
test(sample,riffle5,0.05,">"),
test(sample,riffle6,0.05,">"),
test(sample,riffle7,0.05,">"),
test(sample,riffle8,0.05,">"),
test(sample,riffle9,0.05,">"),
test(sample,riffle10,0.05,">"),
test(sample,riffle11,0.05,">"),
test(sample,riffle12,0.05,">"),
test(sample,riffle13,0.05,">"),
test(sample,riffle14,0.05,">"),
test(sample,riffle15,0.05,">")]

xdf=pd.DataFrame(x)

writer = pd.ExcelWriter('bintests.xlsx', engine='xlsxwriter')
xdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()

sample=[]
riffle1=[]
riffle2=[]
riffle3=[]
riffle4=[]
riffle5=[]
riffle6=[]
riffle7=[]
riffle8=[]
riffle9=[]
riffle10=[]


for i in range(1000):
    sample.append(disorder(random.sample(range(1,53),52))[0])
    deck1 = list(range(1,53))
    deck2 = list(range(1,53))
    deck3 = list(range(1,53))
    deck4 = list(range(1,53))
    deck5 = list(range(1,53))
    deck6 = list(range(1,53))
    deck7 = list(range(1,53))
    deck8 = list(range(1,53))
    deck9 = list(range(1,53))
    deck10 = list(range(1,53))
        
    for i in range(1):
        deck1 = randomrifflecut(deck = deck1)
    for i in range(2):
        deck2 = randomrifflecut(deck = deck2)
    for i in range(3):
        deck3 = randomrifflecut(deck = deck3)
    for i in range(4):
        deck4 = randomrifflecut(deck = deck4)
    for i in range(5):
        deck5 = randomrifflecut(deck = deck5)
    for i in range(6):
        deck6 = randomrifflecut(deck = deck6)
    for i in range(7):
        deck7 = randomrifflecut(deck = deck7)
    for i in range(8):
        deck8 = randomrifflecut(deck = deck8)
    for i in range(9):
        deck9 = randomrifflecut(deck = deck9)
    for i in range(10):
        deck10 = randomrifflecut(deck = deck10)
        
    riffle1.append(disorder(deck1)[0])
    riffle2.append(disorder(deck2)[0])
    riffle3.append(disorder(deck3)[0])
    riffle4.append(disorder(deck4)[0])
    riffle5.append(disorder(deck5)[0])
    riffle6.append(disorder(deck6)[0])
    riffle7.append(disorder(deck7)[0])
    riffle8.append(disorder(deck8)[0])
    riffle9.append(disorder(deck9)[0])
    riffle10.append(disorder(deck10)[0])
    
#sample = pd.DataFrame(sample)
#sns.kdeplot(data=sample)


#sampledf = pd.DataFrame(sample)
#sns.kdeplot(data=sampledf)


x=[test(sample,riffle1,0.05,">"),
test(sample,riffle2,0.05,">"),
test(sample,riffle3,0.05,">"),
test(sample,riffle4,0.05,">"),
test(sample,riffle5,0.05,">"),
test(sample,riffle6,0.05,">"),
test(sample,riffle7,0.05,">"),
test(sample,riffle8,0.05,">"),
test(sample,riffle9,0.05,">"),
test(sample,riffle10,0.05,">")]

xdf=pd.DataFrame(x)

writer = pd.ExcelWriter('bintestcutU1.xlsx', engine='xlsxwriter')
xdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()



sample=[]
riffle1=[]
riffle2=[]
riffle3=[]
riffle4=[]
riffle5=[]
riffle6=[]
riffle7=[]
riffle8=[]
riffle9=[]
riffle10=[]


for i in range(1000):
    sample.append(disorder(random.sample(range(1,53),52))[5])
    deck1 = list(range(1,53))
    deck2 = list(range(1,53))
    deck3 = list(range(1,53))
    deck4 = list(range(1,53))
    deck5 = list(range(1,53))
    deck6 = list(range(1,53))
    deck7 = list(range(1,53))
    deck8 = list(range(1,53))
    deck9 = list(range(1,53))
    deck10 = list(range(1,53))
        
    for i in range(1):
        deck1 = randomrifflecut(deck = deck1)
    for i in range(2):
        deck2 = randomrifflecut(deck = deck2)
    for i in range(3):
        deck3 = randomrifflecut(deck = deck3)
    for i in range(4):
        deck4 = randomrifflecut(deck = deck4)
    for i in range(5):
        deck5 = randomrifflecut(deck = deck5)
    for i in range(6):
        deck6 = randomrifflecut(deck = deck6)
    for i in range(7):
        deck7 = randomrifflecut(deck = deck7)
    for i in range(8):
        deck8 = randomrifflecut(deck = deck8)
    for i in range(9):
        deck9 = randomrifflecut(deck = deck9)
    for i in range(10):
        deck10 = randomrifflecut(deck = deck10)
        
    riffle1.append(disorder(deck1)[5])
    riffle2.append(disorder(deck2)[5])
    riffle3.append(disorder(deck3)[5])
    riffle4.append(disorder(deck4)[5])
    riffle5.append(disorder(deck5)[5])
    riffle6.append(disorder(deck6)[5])
    riffle7.append(disorder(deck7)[5])
    riffle8.append(disorder(deck8)[5])
    riffle9.append(disorder(deck9)[5])
    riffle10.append(disorder(deck10)[5])
    
#sample = pd.DataFrame(sample)
#sns.kdeplot(data=sample)

#sampledf = pd.DataFrame(sample)
#sns.kdeplot(data=sampledf)


x=[test(sample,riffle1,0.05,"<"),
test(sample,riffle2,0.05,"<"),
test(sample,riffle3,0.05,"<"),
test(sample,riffle4,0.05,"<"),
test(sample,riffle5,0.05,"<"),
test(sample,riffle6,0.05,"<"),
test(sample,riffle7,0.05,"<"),
test(sample,riffle8,0.05,"<"),
test(sample,riffle9,0.05,"<"),
test(sample,riffle10,0.05,"<")]

xdf=pd.DataFrame(x)

writer = pd.ExcelWriter('bintestcutr.xlsx', engine='xlsxwriter')
xdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()

sample=[]
riffle1=[]
riffle2=[]
riffle3=[]
riffle4=[]
riffle5=[]
riffle6=[]
riffle7=[]
riffle8=[]
riffle9=[]
riffle10=[]
riffle11=[]
riffle12=[]
riffle13=[]
riffle14=[]
riffle15=[]


for i in range(1000):
    sample.append(disorder(random.sample(range(1,53),52))[6])
    deck1 = list(range(1,53))
    deck2 = list(range(1,53))
    deck3 = list(range(1,53))
    deck4 = list(range(1,53))
    deck5 = list(range(1,53))
    deck6 = list(range(1,53))
    deck7 = list(range(1,53))
    deck8 = list(range(1,53))
    deck9 = list(range(1,53))
    deck10 = list(range(1,53))
    deck11 = list(range(1,53))
    deck12 = list(range(1,53))
    deck13 = list(range(1,53))
    deck14 = list(range(1,53))
    deck15 = list(range(1,53))
        
    for i in range(1):
        deck1 = randomrifflecut(deck = deck1)
    for i in range(2):
        deck2 = randomrifflecut(deck = deck2)
    for i in range(3):
        deck3 = randomrifflecut(deck = deck3)
    for i in range(4):
        deck4 = randomrifflecut(deck = deck4)
    for i in range(5):
        deck5 = randomrifflecut(deck = deck5)
    for i in range(6):
        deck6 = randomrifflecut(deck = deck6)
    for i in range(7):
        deck7 = randomrifflecut(deck = deck7)
    for i in range(8):
        deck8 = randomrifflecut(deck = deck8)
    for i in range(9):
        deck9 = randomrifflecut(deck = deck9)
    for i in range(10):
        deck10 = randomrifflecut(deck = deck10)
    for i in range(11):
        deck11 = randomrifflecut(deck = deck11)
    for i in range(12):
        deck12 = randomrifflecut(deck = deck12)
    for i in range(13):
        deck13 = randomrifflecut(deck = deck13)
    for i in range(15):
        deck14 = randomrifflecut(deck = deck14)
    for i in range(14):
        deck15 = randomrifflecut(deck = deck15)
        
    riffle1.append(disorder(deck1)[6])
    riffle2.append(disorder(deck2)[6])
    riffle3.append(disorder(deck3)[6])
    riffle4.append(disorder(deck4)[6])
    riffle5.append(disorder(deck5)[6])
    riffle6.append(disorder(deck6)[6])
    riffle7.append(disorder(deck7)[6])
    riffle8.append(disorder(deck8)[6])
    riffle9.append(disorder(deck9)[6])
    riffle10.append(disorder(deck10)[6])
    riffle11.append(disorder(deck11)[6])
    riffle12.append(disorder(deck12)[6])
    riffle13.append(disorder(deck13)[6])
    riffle14.append(disorder(deck14)[6])
    riffle15.append(disorder(deck15)[6])
    
#sample = pd.DataFrame(sample)
#sns.kdeplot(data=sample)

#sampledf = pd.DataFrame(sample)
#sns.kdeplot(data=sampledf)


x=[test(sample,riffle1,0.05,">"),
test(sample,riffle2,0.05,">"),
test(sample,riffle3,0.05,">"),
test(sample,riffle4,0.05,">"),
test(sample,riffle5,0.05,">"),
test(sample,riffle6,0.05,">"),
test(sample,riffle7,0.05,">"),
test(sample,riffle8,0.05,">"),
test(sample,riffle9,0.05,">"),
test(sample,riffle10,0.05,">"),
test(sample,riffle11,0.05,">"),
test(sample,riffle12,0.05,">"),
test(sample,riffle13,0.05,">"),
test(sample,riffle14,0.05,">"),
test(sample,riffle15,0.05,">")]

xdf=pd.DataFrame(x)

writer = pd.ExcelWriter('bintestcuts.xlsx', engine='xlsxwriter')
xdf.to_excel(writer, sheet_name='Sheet 1', index=False)
writer.close()
"""



        
    
                  


    
    





