import matplotlib.pyplot as plt

# Read times from the file
with open("times.txt", "r") as file:
    line = file.readline().strip()
    sequential_time, parallel_time, speedup = map(float, line.split())

# Plotting the execution times and speedup
labels = ['Sequential', 'Parallel']
times = [sequential_time, parallel_time]

# Create bar plot for times
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(labels, times, color=['blue', 'orange'])
plt.title('Execution Time')
plt.ylabel('Time (seconds)')
plt.xlabel('Method')
plt.yscale('log')  # Use log scale for better visualization if times differ greatly

# Plot the speedup
plt.subplot(1, 2, 2)
plt.bar(['Speedup'], [speedup], color='green')
plt.title('Speedup (Sequential / Parallel)')
plt.ylabel('Speedup Factor')

# Show the plot
plt.tight_layout()
plt.show()
