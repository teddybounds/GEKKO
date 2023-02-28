#Theodore Bounds
#Multiphase Trajectory Optimization of Bouncing Ball Hitting Target
#Reversed y-velocity variable used to set initial velocity of next phase

###Import Libraries
import math
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.animation as animation
import numpy as np
from gekko import GEKKO

### Inputs
n = 3  # Number of phases/bounces
TX = 10  #target x position (m)
TY = 3   #target y position (m)
TVY = 0  #target y velocity (m)
K = .8 #coefficient of restitution (how much energy is lost per bounce)
vxmax = TX; #relating maximum x-velocity to distance provided best chance of solution
g = 9.81 #define gravity


###Defining Model Parameters
m = GEKKO(remote=True)
nodes = 4 # Two intermediate nodes between collocation points
m.options.NODES = nodes
m.options.SOLVER = 3  # IPOPT
m.options.IMODE = 6  # MPC Direct Collocation
N = 50  #number of time points for each phase
t = np.linspace(0,1,N)
m.time = t

### Define Final time Fixed Variable
TF = [m.FV(.5) for i in range(n)];  #time length variable for each phase
for i in range(n):
    TF[i].STATUS = 1  #make time controllable

### Define Parameters
g = m.Const(value=g) #gravity
K = m.Const(value=K) #coeff of restitution
TX = m.Const(value=TX) #target x-position
TY = m.Const(value=TY) #target y-position
TVY = m.Const(value=TVY) #target y-velocity

###Define final condition vector
final = np.zeros(len(m.time))
final[-1] = 1
final = m.Param(value=final)

### Define State Variables
x = [m.Var(value=0, fixed_initial=False) for i in range(n)]
y = [m.Var(value=0, fixed_initial=False) for i in range(n)]
xdot = [m.Var(value=1, fixed_initial=False) for i in range(n)]
ydot = [m.Var(value=1, fixed_initial=False) for i in range(n)]
nydot = [m.Var(value=-K, fixed_initial=False) for i in range(n)] #defining negative ydot variable

### Set objectives and Constraints (Soft Constraints Required Here)
m.fix_initial(x[0], val=0) #fix the initial x-position at 0
m.Minimize(final*(TY-y[n-1])**2) #Minimize the error to the target y-coord
m.Minimize(final*(TX-x[n-1])**2) #Minimize the error to the target x-coord
m.Minimize(final*(TVY-ydot[n-1])**2) #Minimize the error to the target y-vel
for i in range(n-1):
    m.Minimize(final*y[i]**2) #Minimize the error for bounce occuring at y=0

### Define State Variable Limits and State Space Model
for i in range(n):
    xdot[i].UPPER = vxmax
    y[i].LOWER = 0 #can't be negative
    m.Equation(x[i].dt()/TF[i] == xdot[i])
    m.Equation(y[i].dt()/TF[i] == ydot[i])
    m.Equation(xdot[i].dt()/TF[i] == 0)
    m.Equation(ydot[i].dt()/TF[i] == -g)
    m.Equation(nydot[i] == -K*ydot[i])

### Connect Phases at Endpoints
for i in range(n-1):
    m.Connection(x[i+1], x[i], 1, 'end', 1, 'end')
    m.Connection(x[i+1],'calculated', pos1=1, node1=1)
    m.Connection(y[i+1], y[i], 1, 'end', 1, 'end')
    m.Connection(y[i+1], 'calculated', pos1=1, node1=1)
    m.Connection(xdot[i+1], xdot[i], 1, 'end', 1, 'end')
    m.Connection(xdot[i+1],'calculated', pos1=1, node1=1)
    m.Connection(ydot[i+1], nydot[i], 1, 'end', 1, 'end')  #reverses y-velocity at phase transitions
    m.Connection(ydot[i+1],'calculated', pos1=1, node1=1)

m.solve()

### Display Outputs
#calculate error to target
xerr = np.abs(TX.VALUE-x[n-1].VALUE[-1]) #calculate error to target
yerr = np.abs(TY.VALUE-y[n-1].VALUE[-1])
vyerr = np.abs(TVY.VALUE-ydot[n-1].VALUE[-1])

f_string = f"Target Error is: X_Err = {xerr:.2e}, Y_Err = {yerr:.2e}, and VY_Err = {vyerr:.2e}"
print(f_string)

for i in range(n):
    tfi = round(TF[i].value[0], 2)
    f_string = f"The Duration of Phase {i+1} is {tfi} sec"
    print(f_string)
    bei = y[i].value[-1]
    f_string = f"The Bounce Error of Phase {i+1} is {bei:.2e}"
    if i < n-1: print(f_string)

print("The Total Duration is", round(sum(TF[i].VALUE[0] for i in range(n)), 2), "sec")

########################################
####Plotting the results
import matplotlib.pyplot as plt

### Generate plots

plt.close('all')

fig1 = plt.figure(); ax1 = fig1.add_subplot()
fig2 = plt.figure(); ax2 = fig2.add_subplot()

t = [m.time*TF[i].value[0] for i in range(n)] #Compute full time vector
for i in range(n-1):
    t[i+1] += t[i][-1]

for i in range(n):
    ax1.plot(x[i].value, y[i].value, 'b', lw=2)
    ax2.plot(t[i],ydot[i].value, 'b', lw=2)
    ax2.plot(t[i],nydot[i].value, 'r', lw=2)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Ball Trajectory')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Y-Velocity (m/s)')
ax2.set_title('Ball Vertical Velocity')
ax2.legend(['Ball velocity', 'Negative velocity'])
plt.show()

