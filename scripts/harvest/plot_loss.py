import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True, help="Path to CSV file containing loss data.")
args = parser.parse_args()

df = pd.read_csv(args.csv)
print(df.head())

# head of data:
#    epoch  minibatch          outcome      value
# 0    1.0        100    training loss  94.216507
# 1    1.0        100  validation loss  96.549492
# 2    1.0        200    training loss  75.626625
# 3    1.0        200  validation loss  79.832657
# 4    1.0        300    training loss  71.480400

plt.figure(figsize=(10, 6))
for outcome in df["outcome"].unique():
    subset = df[df["outcome"] == outcome]
    plt.plot(subset["minibatch"], subset["value"], label=outcome)
plt.xlabel("Minibatch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Minibatches")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()