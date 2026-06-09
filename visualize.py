import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\baran\wormtrack\worm_tracks.csv")

print(f"Total frames: {df['timestamp'].nunique()}")
print(f"Total unique worms: {df['worm_id'].nunique()}")
print(f"Average worms per frame: {df.groupby('timestamp')['worm_id'].count().mean():.1f}")

#this section simply plots the trajectories of the worms
plt.figure(figsize=(10, 8))

for worm_id in df['worm_id'].unique():
    worm_data = df[df['worm_id'] == worm_id]
    if len(worm_data) > 10:  #limits worms s.t. those that were in more than 10 frames get plotted
        plt.plot(worm_data['x'], worm_data['y'], alpha=0.5, linewidth=1)
        plt.plot(worm_data['x'].iloc[0], worm_data['y'].iloc[0], 'go', markersize=5)
        plt.plot(worm_data['x'].iloc[-1], worm_data['y'].iloc[-1], 'ro', markersize=5)

plt.title("Worm Trajectories")
plt.xlabel("X position (pixels)")
plt.ylabel("Y position (pixels)")
plt.gca().invert_yaxis()
plt.figtext(0.02, 0.02, "Green = start, Red = end", fontsize=8)
plt.savefig(r"C:\Users\baran\wormtrack\trajectories.png")
plt.show()

print("Saved trajectories.png")
