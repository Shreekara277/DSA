import matplotlib.dates as mdates
import numpy as np
import datetime as dt
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from scipy import special
######################################################
#CONVERTING DATES TO NUMBERS****

dates = np.array(['09/19/1981','03/15/1997','03/29/2000','10/16/2000',
	'03/06/2002','07/11/2003','09/09/1998','1/28/2009',
	'05/28/2004','10/25/2007','04/26/2006','07/08/2014'])

x2 = [[0 for i in range(6)] for i in range(len(dates))]

	
for i in range(len(dates)):
	y2 = dates[i].split('/')
	x2[i][0] = int(y2[0])
	x2[i][1] = int(y2[1])
	x2[i][2] = int(y2[2])
	x2[i][3] = 0
	x2[i][4] = 0
	x2[i][5] = 0
	
def year_fraction(date):
    start = dt.date(date.year, 1, 1).toordinal()
    year_length = dt.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

final = [0 for i in range(len(dates))]
for i in range(len(final)):
	final[i] = year_fraction(dt.datetime(x2[i][2],x2[i][0],x2[i][1],x2[i][3],x2[i][4],x2[i][5]))


final = np.array(final)

#print(final)
########################################
##########ERROR PLOT OF G VS DATES
G = np.array([6.6726,6.6740,6.674255,6.67553,6.67422,6.67387,6.6723,
	6.67346,6.67234,6.67554,6.67455, 6.67191])

G = G*1e-11

sigma_G = np.array([0.0005, 0.0007, 0.000092, 0.00040, 0.00098, 0.00027,
 0.0009, 0.00021, 0.00014, 0.00016,0.00013, 0.00099])

sigma_G = sigma_G*1e-11


###########################################################
######DEFINING MODELS*
def model1(x,A,phi,P,myu):
	return A*np.sin(phi + 2*np.pi*(x/P)) + myu
	
def model2(x,myu):
	return myu*(x**0)

def log_likelihood1(A,phi,P,myu, sigma_sys):
  sigma_GG = np.sqrt(sigma_G ** 2 + sigma_sys ** 2 )
  return np.log(np.sum( (1/(sigma_GG*np.sqrt(2*np.pi)) * np.exp( ((model1(final,A,phi,P,myu) - G)/(np.sqrt(2)*sigma_GG))**2))))

def log_likelihood2(myu, sigma_sys):
  sigma_GG = np.sqrt(sigma_G ** 2 + sigma_sys ** 2 )
  return np.log(np.sum( (1/(sigma_GG*np.sqrt(2*np.pi)) * np.exp( ((model2(final,myu) - G)/(np.sqrt(2)*sigma_GG))**2))))

def chi2_1(A,phi,P,myu, sigma_sys):
  sigma_GG = np.sqrt(sigma_G ** 2 + sigma_sys ** 2)
  return np.sum(( (model1(final,A,phi,P,myu) - G)/(sigma_GG))**2)
	
def chi2_2(myu, sigma_sys):
  sigma_GG = np.sqrt(sigma_G ** 2 + sigma_sys ** 2)
  return np.sum((( (model2(final,myu) - G)/(sigma_GG))**2))

def P(c, nu):
  return (c ** (nu/2 - 1) * np.exp(- c/2)) / (2 ** (nu/2) * special.gamma(nu/2))

def AIC(l, k, N):
  return (-2 * l + k * np.log(N))

def BIC(l,k):
  return (-2 * l + 2 * k)
#############
##############HYPOTHESII

popt,pcov = curve_fit(model1,final,G,sigma=sigma_G)#crve
print(popt)
H1 = 6.6741*1e-11
H2 = np.array([1.64*1e-14, -0.07, 5.9, 6.674*1e-11])
H3 = np.array([1.9*1e-14, 0.0011, 7.57, 6.571*1e-11])
##################
########chi2

chi24 = chi2_1(popt[0],popt[1],popt[2],popt[3],0)
#chi21 = chi2_2(H1)
chi21 = chi2_2(H1, 1e-14)
chi22 = chi2_1(H2[0],H2[1],H2[2],H2[3],0)
chi23 = chi2_1(H3[0], H3[1], H3[2], H3[3], 1e-12)
#chi24 = chi2_1(H4[0],H4[1],H4[2],H4[3])

pp = [P(chi24, 8), P(chi21, 10), P(chi22, 8), P(chi23, 7)]
print(pp)
chisquare = [chi24/8,chi21/10,chi22/8, chi23/7]
print(chisquare)
ll = [log_likelihood1(popt[0],popt[1],popt[2],popt[3],0), log_likelihood2(H1, 1e-14), log_likelihood1(H2[0],H2[1],H2[2],H2[3],0),
      log_likelihood1(H3[0], H3[1], H3[2], H3[3], 1e-12)]

aic = [AIC(ll[0], 4, 12), AIC(ll[1], 2, 12), AIC(ll[2], 4, 12), AIC(ll[3], 5, 12)]
bic = [BIC(ll[0], 4), BIC(ll[1], 2), BIC(ll[2], 4), BIC(ll[3], 5)]
print(aic - aic[1],'\n', bic - bic[1])

########################
########PLOTTING THE MODEL
plt.figure(1)
#plt.plot(final,G,'o')
plt.errorbar(final, G, yerr=sigma_G,fmt=".m", capsize = 2,  label = 'Obtained data set')


x0 = np.linspace(1980,2015,10000)
y0 = model1(x0,popt[0],popt[1],popt[2],popt[3])
y1 = model1(x0,H2[0],H2[1],H2[2],H2[3])
y2 = model2(x0,H1)

plt.plot(x0,y0, label = 'Model derived from curve-fit')
plt.plot(x0,y2, label = 'Constant Model')
plt.xlabel("dates")
plt.ylabel("Obtained value of G")
plt.legend()
print("\n \n")
plt.figure(2)
plt.errorbar(final, G, yerr=sigma_G,fmt=".m", capsize = 2, label = 'Obtained data set')
plt.plot(x0,y1, label = 'Hypothesis 3')
plt.plot(x0,y2, label = 'Hypothesis 2')
plt.xlabel("dates")
plt.ylabel("Obtained value of G")
plt.legend()


plt.show()


