import math
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
st.set_page_config(page_title="Régression linéaire multiple",layout="wide")
st.title("Régression linéaire multiple")
ouinon=["oui","non"]
f0=lambda x:x**0
f1=lambda x:x**1
f2=lambda x:x**2
f3=lambda x:x**3
f4=lambda x:x**4
f5=lambda x:x**5
f6=lambda x:x**6
f7=lambda x:x**7
f8=lambda x:x**8
f9=lambda x:x**9
f10=lambda x:x**10
f11=lambda x:x**11
f12=lambda x:x**12
f13=lambda x:x**13
f14=lambda x:x**14
f15=lambda x:x**15
f16=lambda x:x**16
f17=lambda x:x**17
f18=lambda x:x**18
f19=lambda x:x**19
f20=lambda x:x**20
poly=np.array((f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20))


#vecteur de fonctions a prendre en compte a remplir en fonction des envie
pre=np.array((math.sin,math.cos,math.tan,math.exp,math.log))

funct=st.multiselect("Choisir les fonctions polynomiales",options=[f"sin(x)",f"cos(x)",f"tan(x)",f"exp(x)",f"log(x)"]+[f"x^{i}" for i in range(21)]+[f"1/x^{i}" for i in range(1,21)],default=["x^0","x^1"],key="poly")

f=[]
for func in funct:
      if func.startswith("x^"):
        deg=int(func[2:])
        f.append(poly[deg])
      elif func=="sin(x)":
        f.append(math.sin)
      elif func=="cos(x)":
        f.append(math.cos)
      elif func=="tan(x)":
        f.append(math.tan)
      elif func=="exp(x)":
        f.append(math.exp)
      elif func=="log(x)":
        f.append(math.log)
      elif func.startswith("1/x^"):
        deg=int(func[5:])
        f.append(lambda x,deg=deg: 1/poly[deg](x))




#lecture des données
data_input=st.expander("Input des données")

X_file=data_input.checkbox("importer les données depuis un fichier CSV",key="file")
if X_file:
    uploaded_file=data_input.file_uploader("choisir un fichier CSV",type=["csv"],key="upfile")
    if uploaded_file is not None:
        data=pd.read_csv(uploaded_file)
        data_input.dataframe(data)
        X=data.iloc[:,0].to_numpy()
        Y=data.iloc[:,1].to_numpy()
else:
    X_lin=data_input.number_input("Nombre de lignes",min_value=1,value=3,step=1,key="nlinX")
    X=data_input.data_editor(pd.DataFrame([0]*X_lin),key="data")


Y_file=data_input.checkbox("importer les données depuis un fichier CSV",key="fileY")
if Y_file:
    uploaded_file=data_input.file_uploader("choisir un fichier CSV",type=["csv"],key="upfileY")
    if uploaded_file is not None:
        data=pd.read_csv(uploaded_file)
        data_input.dataframe(data)
        X=data.iloc[:,0].to_numpy()
        Y=data.iloc[:,1].to_numpy()
else:
    Y=data_input.data_editor(pd.DataFrame([i for i in range(len(X))]),key="dataY")

X=X.to_numpy()
Y=Y.to_numpy()
X=np.sort(X)
Y=np.sort(Y)


#regression linéaire multiple
J=np.zeros((len(X),len(f)))
for i in range(len(X)):
      for j in range(len(f)):
            J[i,j]=f[j](X[i])
Jt=np.transpose(J)

#Beta=np.linalg.inv(Jt@J)@(Jt@Y) marche pas pour des problèmes de conditionnement

Beta=np.linalg.solve(Jt@J,Jt@Y)

print(np.round(Beta,4))
SCT=np.sum((Y-np.mean(Y))**2)
SCR=np.sum((np.mean(Y)-J@Beta)**2)
SCE=np.sum((Y-J@Beta)**2)


p=len(f)
n=len(X)

print("R²=",SCR/SCT)
print("Ra²=",1-(n-1)/(n-p)*SCE/SCT,"\n")

Sigmac=SCE/(n-p)
#Analyse gaussienne
#décision du alpha
param=st.expander("Paramètres statistiques")
alpha=param.slider("Choisir le niveau de confiance",min_value=0.80,max_value=0.99,value=0.95,step=0.01,key="alpha")

param.write("".join([str(Beta[i][0])+"\*"+funct[i]+" + " for i in range(len(Beta))]).strip("+ "))
#----------------------------------------
param.write("R²="+str(round(SCR/SCT,4)))
param.write("Ra²="+str(round(1-(n-1)/(n-p)*SCE/SCT,4))+"\n\n")
param.divider()
if np.linalg.det(Jt@J)!=0:
    gam=np.linalg.inv(Jt@J)
    for i in range(p):
        ICi=np.round(Beta[i]-math.sqrt(Sigmac*gam[i,i])*scipy.stats.t.ppf(q=1-alpha,df=n-p),6)
        ICs=np.round(Beta[i]+math.sqrt(Sigmac*gam[i,i])*scipy.stats.t.ppf(q=1-alpha,df=n-p),6)
        param.write("l'intervalle de confiance pour la "+str(i)+" eme composante")
        param.write("["+str(ICi[0])+" ; "+str(ICs[0])+"]")
        param.write("la "+str(i)+" eme composante est-elle nulle ? : "+str(ouinon[int(ICs[0]*ICi[0]>0)])+"\n\n")
        param.divider()
    Model=np.sum(Beta)/np.ones(np.shape(Beta)).transpose()@gam@np.ones(np.shape(Beta))*np.sum(Beta)/Sigmac
    param.write("le modèle est inutile ? : "+str(ouinon[int(Model>scipy.stats.f.ppf(q=1-alpha,dfn=1,dfd=n-p))])+"\n\n") 
    

#affichage graphique
ecart=np.arange(min(X),max(X),0.5)
V=np.zeros((len(ecart),len(f)))
for i in range(len(ecart)):
      for j in range(len(f)):
            V[i,j]=f[j](ecart[i])
aff=np.array(V)@Beta
fig,ax=plt.subplots()
ax.plot(X,Y,".")
ax.plot(ecart,aff)
st.pyplot(fig,width=700)
