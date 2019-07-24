import pickle
import matplotlib.pyplot as plt

with open('disc_loss.pkl', 'rb') as f:
    cost_hist = pickle.load(f)

with open('gen_loss.pkl', 'rb') as f:
     acc_hist = pickle.load(f)

plt.plot(list(range(len(cost_hist))), cost_hist)
plt.title("loss of discriminator")
plt.savefig('loss1.png')
plt.clf()

print("plotting change in accuracy")
# Plotting accuracy

plt.plot(list(range(len(acc_hist))), acc_hist)
plt.title("loss of generator")
plt.savefig('loss2.png')

print("Done!")
