import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../.logs/tiered/MAML_meta_loss_log.csv')
plt.figure(figsize=(14, 6))
plt.plot(df['step'], df['meta_loss'], color='blue', alpha=0.2, label='Raw Meta Loss', linewidth=0.5)

window_size = 500  
df['moving_avg'] = df['meta_loss'].rolling(window=window_size).mean()

plt.plot(df['step'], df['moving_avg'], color='red', label=f'Moving Average (window={window_size})', linewidth=2)

plt.xlabel('Step')
plt.ylabel('Meta Loss')
plt.title('MAML Meta Loss over 60.000 steps')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig("tiered_training.png")
