"""Shared oblivious network, executed with clear scalars in correctness tests.

The same network is imported by the MPC source with a secret select callback.
All loop bounds and memory addresses depend only on public list lengths.
"""


def bitonic(records, sentinel, select):
    n=1
    while n<len(records): n*=2
    a=list(records)+[list(sentinel) for _ in range(n-len(records))]
    k=2
    while k<=n:
        j=k//2
        while j:
            for i in range(n):
                other=i^j
                if other>i:
                    swap=(a[i][0]>a[other][0]) if (i&k)==0 else (a[i][0]<a[other][0])
                    left,right=a[i],a[other]
                    a[i]=[select(swap,y,x) for x,y in zip(left,right)]
                    a[other]=[select(swap,x,y) for x,y in zip(left,right)]
            j//=2
        k*=2
    return a


def matvec(entries, vector, output_size, select):
    """Entries (destination, source, value), already padded with zero values."""
    m=len(entries); n=len(vector)
    sentinel=[4*(m+n+output_size+1)**2,0,0,0,0]
    records=[[2*j,0,value,0,j] for j,value in enumerate(vector)]
    records += [[2*source+1,dest,value,1,n+i] for i,(dest,source,value) in enumerate(entries)]
    gathered=bitonic(records,sentinel,select)
    carry=0
    products=[]
    for _,dest,value,kind,identifier in gathered:
        carry=select(kind==0,value,carry)
        product=value*carry
        # Vector and padding records become zero contributions at destination 0.
        products.append([2*dest, dest, select(kind==1,product,0),0,identifier])
    products += [[2*j+1,j,0,1,len(gathered)+j] for j in range(output_size)]
    reduced=bitonic(products,sentinel,select)
    accum=0; previous=-1; outputs=[]
    for _,dest,value,marker,identifier in reduced:
        accum=select(dest==previous,accum,0)+value
        previous=dest
        key=select(marker==1,dest,output_size+identifier)
        outputs.append([key,0,accum,0,0])
    ordered=bitonic(outputs,sentinel,select)
    return [r[2] for r in ordered[:output_size]]
