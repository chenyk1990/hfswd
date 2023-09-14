from pylib.io import asciiread
import numpy as np

lines1=asciiread('evetmp/eveHF2.dat')
lines2=asciiread('evetmp/eveHFef.dat')
lines=lines1+lines2
lines=np.array(lines)
np.save('txed_evehf.npy',lines)



lines=asciiread("evetmp/eveSWDsh.dat")
lines=np.array(lines)
np.save('txed_eveSWDsh.npy',lines)

lines=asciiread("evetmp/eveSWDdp.dat")
lines=np.array(lines)
np.save('txed_eveSWDdp.npy',lines)




