from fractions import Fraction as F
import itertools, math
EPS=F(2748779,274877906944); G=F(11388943,524288); B2=F(3172863,33554432); SG=F(7614871,33554432); OUT=F(2887801,65536)
def mu(t): return F(2,24) if t<5 else (F(-1,24) if t==5 else F(t-5,24))
def Q(t): return F(452) if t<5 else (F(451) if t==5 else 452+F(t-5)**2)
def sd2(t): return Q(t)/24-mu(t)**2+EPS
# rational sqrt bounds with 4 decimals
def sqb(x, q=F(1,10000)):
    v=math.sqrt(float(x)); lo=F(math.floor(v/float(q)))*q; hi=lo+q
    while lo*lo> x: lo-=q
    while hi*hi< x: hi+=q
    return lo,hi
SD={t:sqb(sd2(t)) for t in [0,5,6,7,8,9]}
R24=sqb(F(24),F(1,100000))
print('sd bounds',{t:(float(a),float(b)) for t,(a,b) in SD.items()}, 'sqrt24',R24)
def nuabs(t): return (1-mu(t)) if t<5 else (1+mu(t))   # |mk - mu|
# Delta_{k,m} = 100*|nu_n|/sqrt24 * (|nu_0| + |nu_{5+m}|), nu_t = nuabs(t)/sd(t)
def Dbounds(k,m):
    n=5+k
    lo=100*nuabs(n)/(SD[n][1]*R24[1])*(nuabs(0)/SD[0][1]+nuabs(5+m)/SD[5+m][1])
    hi=100*nuabs(n)/(SD[n][0]*R24[0])*(nuabs(0)/SD[0][0]+nuabs(5+m)/SD[5+m][0])
    return lo,hi
GRID={0:(F(195,100),F(2)),1:(F(21,10),F(225,100)),2:(F(215,100),F(235,100)),3:(F(225,100),F(25,10)),4:(F(23,10),F(26,10))}
for k in range(5):
    for m in range(k+1):
        lo,hi=Dbounds(k,m); assert GRID[k][0]<=lo and hi<=GRID[k][1],(k,m,float(lo),float(hi))
print('Delta grid ok')
def P(y): return 1+y+y**2/2+y**3/6+y**4/24
def expneg_upper(q): y=q/4; return 1/P(y)**4        # exp(-q) <= this
def expneg_lower(q): y=q/4; return 1/(P(y)+y**5*6/(120*5))**4   # exp(-q) >= this
def dec_down(x,d=4): return F(math.floor(x*10**d),10**d)
def dec_up(x,d=4): return F(math.ceil(x*10**d),10**d)
RL={k:dec_down(expneg_lower(GRID[k][1])) for k in range(5)}
RH={k:dec_up(expneg_upper(GRID[k][0])) for k in range(5)}
print('r bounds',{k:(float(RL[k]),float(RH[k])) for k in range(5)})
# alpha = G/(sd0 (5+R)), g_m = G r_m/(sd_{5+m}(5+R)),  R in [(k+1)RL,(k+1)RH]
AL={};AH={};GL={};GH={}
for k in range(5):
    AL[k]=dec_down(G/(SD[0][1]*(5+(k+1)*RH[k]))); AH[k]=dec_up(G/(SD[0][0]*(5+(k+1)*RL[k])))
    for m in range(k+1):
        GL[k,m]=dec_down(G*RL[k]/(SD[5+m][1]*(5+(k+1)*RH[k]))); GH[k,m]=dec_up(G*RH[k]/(SD[5+m][0]*(5+(k+1)*RL[k])))
    print('k',k,'alpha',(float(AL[k]),float(AH[k])),'g',[(float(GL[k,m]),float(GH[k,m])) for m in range(k+1)])
S2L,S2H=F(432,100),F(454,100)
TH={0:(F(30,100),F(96,100)),1:(F(39,100),F(99,100)),2:(F(395,1000),F(975,1000)),3:(F(315,1000),F(905,1000)),4:(F(195,1000),F(80,100))}
def muO(m): return F(-1,24) if m==0 else F(m,24)
for k in range(5):
    tl,th=TH[k]; worst_lo=-10**9; worst_hi=10**9
    for verts in itertools.product((0,1),repeat=k+3):
        a=(AL[k],AH[k])[verts[0]]; s2=(S2L,S2H)[verts[1]]; g={m:(GL[k,m],GH[k,m])[verts[2+m]] for m in range(k+1)}
        for e in range(10):
            for c in range(6):
                Z=a*(c-F(5*(e+1),12))+sum(g[m]*((1 if 1<=m<=c else 0)-(e+1)*muO(m)) for m in range(k+1))-k+(e+1)*B2*s2
                if c<=k: worst_lo=max(worst_lo, Z-tl*SG*s2)
                else: worst_hi=min(worst_hi, Z-th*SG*s2)
    rt=OUT*SG*(th-tl); ro=1+OUT*SG*max(tl,1-th)
    print('k',k,'lo slack',float(-worst_lo),'hi slack',float(worst_hi),'final',float(rt),float(ro), rt>ro and worst_lo<0 and worst_hi>0)
# GO_9 bounds
for k in range(5):
    vals=[]
    for verts in itertools.product((0,1),repeat=k+2):
        a=(AL[k],AH[k])[verts[0]]; g={m:(GL[k,m],GH[k,m])[verts[1+m]] for m in range(k+1)}
        vals.append(a*(5-F(50,12))+sum(g[m]*((1 if 1<=m<=5 else 0)-10*muO(m)) for m in range(k+1)))
    print('GO9',k,float(min(vals)),float(max(vals)))
# s2 checks: lower needs 451/24 - (S2/24)^2 + EPS >= 4.32^2 with |S2|<=7 ; upper Qp/24+23/24+EPS <= 4.54^2
print('s2 lo', float(F(451,24)-F(7,24)**2), float(S2L**2), ' s2 hi', float(F(468+23,24)+EPS), float(S2H**2))
for t in [0,5,6,7,8,9]:
    dg = 0 if t==5 else 1
    num=dg*(1-2*mu(t))+10*mu(t)**2
    print('jensen',t,float(G**2*num/sd2(t)))
