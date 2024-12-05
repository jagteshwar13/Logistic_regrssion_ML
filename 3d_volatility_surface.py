import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define strike prices and time-to-maturity values
strike_price = np.linspace(50, 150, 25)
time = np.linspace(0.5, 2, 25)

# Create a grid for strike prices and time
strike_price, time = np.meshgrid(strike_price, time)

# Calculate implied volatility
implied_vol = (strike_price - 100)**2 / (100 * strike_price * time)

# Create the figure and a 3D axis
fig = plt.figure(figsize=(12, 5))
axis = fig.add_subplot(111, projection="3d")  # Correct way to specify 3D projection

# Plot the surface
surface = axis.plot_surface(strike_price, time, implied_vol, rstride=1, cstride=1, 
                             cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=False)

# Set axis labels
axis.set_xlabel("Strike Price")
axis.set_ylabel("Time to Maturity")
axis.set_zlabel("Implied Volatility")

# Add a color bar
fig.colorbar(surface, shrink=0.5, aspect=5)

# Show the plot
plt.show()



