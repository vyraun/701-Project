import matplotlib.pyplot as plt 

f = plt.figure()

test_scores = [50.005, 56.0 , 53.525, 59.995, 62.986, 63.046, 56.695, 73.5, 77.5, 78.8, 70.0, 80.0, 88.5]

plt.axis([-0.25, 13*0.5 - 0.25, 45, 100])

for i, x in enumerate(test_scores):
    
    if i<0:
        color='k'
    elif i < 7:
        color='b'
        label = 'Semantic Textual Similarity based'
    elif i<10:
        color='r'
        label ='Shallow Classification'
    elif i<12:
        color='m'
        label = 'Deep Classification'
    else:
        color='g'
        label = 'State-of-the-art Accuracy'
    
    if i == 6 or i == 9 or i==11 or i ==12: 
        plt.plot(i*0.5, test_scores[i],  '.', ms =10, color=color, label=label)
    else:
        plt.plot(i*0.5, test_scores[i],  '.', ms =10, color=color)


plt.text(0, 50.005, "Dot+LR", horizontalalignment='left',  fontsize=8) #, horizontalalignment='center',
plt.text(1*0.5, 56.0, "Subspace+LR", horizontalalignment='left',  fontsize=8)
plt.text(2*0.5, 53.525, "Concat+LR", horizontalalignment='left',  fontsize=8)
plt.text(3*0.5, 59.995, "Concat+LR+B", horizontalalignment='left',  fontsize=8)
plt.text(4*0.5, 62.986, "Concat+DT", horizontalalignment='right',  fontsize=8)
plt.text(5*0.5, 63.046, "Concat+DT+B", horizontalalignment='left',  fontsize=8)
plt.text(6*0.5, 56.695, "Concat+Boosting", horizontalalignment='left',  fontsize=8)
plt.text(7*0.5, 73.5, "FT+Uni", horizontalalignment='left',  fontsize=8)
plt.text(8*0.5, 77.5, "FT+Bi", horizontalalignment='left',  fontsize=8)
plt.text(9*0.5, 78.8, "FT+Tri", horizontalalignment='left',  fontsize=8)
plt.text(10*0.5, 70.0, "Siamese LSTM", horizontalalignment='left',  fontsize=8)
plt.text(11*0.5, 80.0, "BiMPM", horizontalalignment='left',  fontsize=8)
plt.text(12*0.5, 89.06, "SoTA", horizontalalignment='center',  fontsize=8)


#plt.axhline(y=88.5, color='y', linestyle='--')

plt.xlabel("Models")
plt.ylabel("Test Accuracies (%)")
plt.title("Preliminary Experiments")
plt.xticks([])
#plt.grid(True)
plt.legend()

f.savefig("foo.pdf", bbox_inches='tight')

plt.show()