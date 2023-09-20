import numpy as np
def gomory(filename):
    with open(filename) as f:
        lines = f.readlines()
        n=int(lines[0][0])
        m=int(lines[0][2])
        c=np.zeros(n)
        b=(lines[1].split())
        i=0
        while i<m:
            b[i]=float(b[i])
            i=i+1
        c=lines[2].split()
        i=0
        while i<n:
            c[i]=-float(c[i])
            i=i+1
        A=np.zeros((m,n))
        for i in range(m):
            A[i]=lines[i+3].split()
        for i in range(m):
            for j in range (n):
                A[i][j]=float(A[i][j])
        A_tableau=np.concatenate((A,np.identity(m)),axis=1)
        first_row=np.concatenate((np.array(c).reshape(1,n),np.zeros(m).reshape(1,m)),axis=1)
        rest_of_tableau=np.concatenate((first_row.reshape(1,n+m),A_tableau),axis=0)
        first_column=np.concatenate((np.array(float(0)).reshape(1,1),np.array(b).reshape(m,1)),axis=0)
        tableau=np.concatenate((first_column,rest_of_tableau),axis=1)
        j=0
        bases=[]
        while j<m:
            bases.append(j+n)
            j=j+1
        [tableau_dual,bases_dual,x_dual]=simplex(tableau,bases)
        
        Out = gomory_cut(tableau_dual,bases_dual,x_dual)[0:n]
        for i in range(0, n):
            Out[i] = round((Out[i]),0)
        return (Out)
                
def simplex(tableau,bases):
    m=np.shape(tableau)[0]
    l=np.shape(tableau)[1]
    reduced_costs=tableau[0][1:l+1]
    entering_index=None
    i=0
    feasible=0
    while i<l-1 and feasible==0:
        if reduced_costs[i]<0 and feasible==0:
            entering_index=i+1
            feasible=1
        else:
            i=i+1
    if entering_index==None:
        x=np.zeros(l-1)
        i=0
        while i<m-1:
            x[int(bases[i])]=tableau[i+1][0]
            i=i+1
        return [tableau,bases,x]
    else:
        entering_column=np.zeros(m-1)
        entering_column=entering_column.reshape(m-1,1)
        k=0
        positive=1
        while k<m-1:
            if tableau[k+1][entering_index]<=0:
                entering_column[k][0]=0
                k=k+1
            else:
                entering_column[k][0]=tableau[k+1][entering_index]
                positive=0
                k=k+1
        i=0
        leaving_index=None
        min=np.inf
        while i<m-1:
            if entering_column[i][0]==0:
                i=i+1
            else:
                if leaving_index==None:
                    leaving_index=i+1
                    min=tableau[i+1][0]/tableau[i+1][entering_index]
                    i=i+1
                else:
                    if tableau[i+1][0]/tableau[i+1][entering_index]<min:
                        leaving_index=i+1
                        min=tableau[leaving_index][0]/tableau[leaving_index][entering_index]
                        i=i+1
                    else:
                        i=i+1
        i=0
        while i<m:
            if i==leaving_index:
                i=i+1
            else:
                tableau[i]=tableau[i]-(tableau[i][entering_index]/tableau[leaving_index][entering_index])*tableau[leaving_index]
                i=i+1
        tableau[leaving_index]=tableau[leaving_index]/tableau[leaving_index][entering_index]
        bases[leaving_index-1]=entering_index-1
        return simplex(tableau,bases)

def dual_simplex(tableau,bases):
    m=np.shape(tableau)[0]
    l=np.shape(tableau)[1]
    first_column=tableau[1:m,0]
    entering_index=None
    i=0
    feasible=0
    while i<m-1 and feasible==0:
        if first_column[i]<0 and feasible==0:
            entering_index=i+1
            feasible=1
        else:
            i=i+1
    if entering_index==None:
        x=np.zeros(l-1)
        i=0
        while i<m-1:
            x[int(bases[i])]=tableau[i+1][0]
            i=i+1
        return [tableau,bases,x]
    else:
        exiting_row=np.zeros(l-1).reshape(1,l-1)
        k=0
        while k<l-1:
            if tableau[entering_index][k+1]>=0:
                exiting_row[0][k]=0
                k=k+1
            else:
                exiting_row[0][k]=tableau[entering_index][k+1]
                k=k+1
        i=0
        leaving_index=None
        min=np.inf
        while i<l-1:
            if exiting_row[0][i]==0:
                i=i+1
            else:
                if leaving_index==None:
                    leaving_index=i+1
                    min=tableau[0][i+1]/abs(tableau[entering_index][i+1])
                    i=i+1
                else:
                    if tableau[0][i+1]/abs(tableau[entering_index][i+1])<min:
                        leaving_index=i+1
                        min=tableau[0][leaving_index]/abs(tableau[entering_index][leaving_index])
                        i=i+1
                    else:
                        i=i+1
        i=0
        while i<m:
            if i==entering_index:
                i=i+1
            else:
                tableau[i]=tableau[i]-(tableau[i][leaving_index]/tableau[entering_index][leaving_index])*tableau[entering_index]
                i=i+1
        tableau[entering_index]=tableau[entering_index]/tableau[entering_index][leaving_index]
        bases[entering_index-1]=leaving_index-1
        return dual_simplex(tableau,bases)
def gomory_cut(tableau_dual,bases_dual,x_dual):
    p=np.shape(tableau_dual)[0]
    q=np.shape(tableau_dual)[1]
    j=1
    non_integer=[]
    max=-np.inf
    max_index=None
    while j<p:
        if tableau_dual[j][0]%1<1e-7 or tableau_dual[j][0]%1>0.999999:
            j=j+1
        else:
            if tableau_dual[j][0]%1>max:
                max=tableau_dual[j][0]%1
                max_index=j
                j=j+1
            else:
                j=j+1
    if max_index!=None:
        new_row=np.zeros(q)
        i=0
        while i<q:
            new_row[i]=-1*(tableau_dual[max_index][i]%1)
            i=i+1
        tableau_dual=np.concatenate((tableau_dual,new_row.reshape(1,q)),axis=0)
        new_column=np.concatenate((np.zeros(p).reshape(p,1),np.array(float(1)).reshape(1,1)),axis=0).reshape(p+1,1)
        tableau_dual=np.concatenate((tableau_dual,new_column),axis=1)
        bases_dual.append(q-1)
        [tableau_dual2,bases_dual2,x_dual2]=dual_simplex(tableau_dual,bases_dual)
        return gomory_cut(tableau_dual2,bases_dual2,x_dual2)
    else:
        return(x_dual)