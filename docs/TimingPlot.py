import matplotlib.pyplot as plt

# Timed on Matlab 2022 on intel i7, 8-core 3.8GHz 

# Preprocessing plot
m = [149.6, 377.5,1352.2]  # matlab timing. time the full constructor phase.
p = [166.8, 904.4,4810.4]  # Python 3.10 on same PC specs. 


N = [50, 100,200]
                # n=50 iterates over two levels.
                # n= 100 has three levels
                # n= 200  has four levels


plt.plot(N, p,marker='x', linestyle='', label='Python Implementation')
plt.plot(N, m, marker='o', linestyle='', label='Matlab Implementation')

plt.xticks([50,100,200])  # Set the positions of the ticks
plt.gca().set_xticklabels([50,100,200])  # Set the labels of the ticks
plt.xlim(00, 250)  # Adjust the limits of the x-axis to control the width

plt.xlabel('Sqrpoisson Data set of Size N', fontsize=20)
plt.ylabel('Runtime (ms)', fontsize=20)
plt.title('Preprocessing Phase', fontsize=20)
plt.legend(fontsize=18)
plt.legend(loc='upper left')


plt.savefig('TimePreprocessing.png', dpi=300)  # Save as PNG format with 300 dpi resolution
plt.show()

plt.clf()
#
# Solver Phase plot
m1 = [31.0,44.2,91.3]  # matlab timing. time the full solving phase.
p1 = [193.2, 1313.8,8398.9]



plt.plot(N, p1,marker='x', linestyle='', label='Python Implementation')
plt.plot(N, m1, marker='o', linestyle='', label='Matlab Implementation')

plt.xticks([50,100,200])  # Set the positions of the ticks
plt.gca().set_xticklabels([50,100,200])  # Set the labels of the ticks
plt.xlim(00, 250)  # Adjust the limits of the x-axis to control the width

plt.xlabel('Sqrpoisson Data set of Size N', fontsize=20)
plt.ylabel('Runtime (ms)', fontsize=20)
plt.title('Solver Phase', fontsize=20)
plt.legend(fontsize=18)
plt.legend(loc='upper left')


plt.savefig('TimeSolver.png', dpi=300)  # Save as PNG format with 300 dpi resolution
plt.show()