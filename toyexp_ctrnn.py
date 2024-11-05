import ctrnn
import matplotlib.pyplot as plt
import numpy as np

# EXPERIMENT PARAMETERS
size = 10
duration = 100
stepsize = 0.01
time = np.arange(0.0,duration,stepsize)

# DEFINITION OF THE OBSERVABLE
# def activity():
#     nn = ctrnn.CTRNN(size)
#     nn.randomizeParameters()

#     # Set some correct proportion of excitatory/inhibitory connections
#     # for i in range(size):
#     #     for j in range(size):
#     #         if np.random.random() < er:
#     #             nn.Weights[i,j] = np.random.random()*10.0 - 0.5
#     #         else:
#     #             nn.Weights[i,j] = 0.0 #np.random.random()*(-10.0)

#     nn.initializeState(np.zeros(size))
#     outputs = np.zeros((len(time),size))

#     # Run transient
#     for t in time:
#         nn.step(stepsize)

#     # Run simulation
#     step = 0
#     for t in time:
#         nn.step(stepsize)
#         outputs[step] = nn.Outputs
#         step += 1

#     # Sum the absolute rate of change of all neurons across time as a proxy for "active"
#     activity = np.sum(np.abs(np.diff(outputs,axis=0)))/(duration*size*stepsize)
#     return activity

# ITERATE THROUGH DIFFERENT PROPORTIONS
reps = 100
# steps = 11
# errange = np.linspace(0.0,1.0,steps)

# data = np.zeros((steps,reps))
# k = 0
# for er in errange:
#     for r in range(reps):
#         data[k][r] = activity(er)
#     k += 1
res = []
for i in range(6):
    act = []
    if i == 0:
        a = 'Sigmoid'
    elif i == 1:
        a = 'ReLU'
    elif i == 2:
        a = 'Leaky ReLU'
    elif i == 3:
        a = 'Tanh'
    elif i == 4:
        a = 'Step'
    elif i == 5:
        a = 'Sine'
    for k in range(reps):
        nn = ctrnn.CTRNN(size, i)
        nn.randomizeParameters()
        nn.initializeState(np.zeros(size))
        outputs = np.zeros((len(time),size))
        for t in time:
          nn.step(stepsize)
         
        step = 0
        for t in time:
            nn.step(stepsize)
            outputs[step] = nn.Outputs
            step += 1
            
        activity = np.sum(np.abs(np.diff(outputs,axis=0)))/(duration*size*stepsize)
        act.append(activity)
    res.append((a, np.mean(act), np.std(act)))

# visualize the results
# plt.plot(errange,np.mean(data,axis=1),'ko')
# plt.plot(errange,np.mean(data,axis=1)+np.std(data,axis=1)/np.sqrt(reps),'k.')
# plt.plot(errange,np.mean(data,axis=1)-np.std(data,axis=1)/np.sqrt(reps),'k.')
labels, means, stds = zip(*res)
print(labels)

plt.bar(labels, means, yerr=stds, capsize=5)
# plt.xlabel("Proportion of excitatory connections")
# plt.ylabel("Amount of activity in the circuit")
# plt.title("Leaky ReLU activation function")
# plt.show()
plt.xlabel("Activation Function")
plt.ylabel("Activity")
plt.title("Effect of Activation Function on Active Neurons")
plt.show()
