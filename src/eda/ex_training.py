import pandas as pd
import matplotlib.pyplot as plt
import io

df = pd.read_csv("../.logs/tiered/MAML_train_log.csv")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# sup_loss
axs[0, 0].plot(df['Step'], df['Pre_Sup_Loss'], color='red', linestyle='--', label='Pre_Sup_Loss', marker='o')
axs[0, 0].plot(df['Step'], df['Post_Sup_Loss'], color='blue', linestyle='--', label='Post_Sup_Loss', marker='o')
axs[0, 0].set_title('Sup_Loss')
axs[0, 0].set_xlabel('Step')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()
axs[0, 0].grid(True)

# que_loss
axs[0, 1].plot(df['Step'], df['Pre_Que_Loss'], color='red', linestyle='-', label='Pre_Que_Loss', marker='o')
axs[0, 1].plot(df['Step'], df['Post_Que_Loss'], color='blue', linestyle='-', label='Post_Que_Loss', marker='o')
axs[0, 1].set_title('Que_Loss')
axs[0, 1].set_xlabel('Step')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[0, 1].grid(True)

# sup_acc
axs[1, 0].plot(df['Step'], df['Pre_Sup_Acc'], color='red', linestyle='--', label='Pre_Sup_Acc', marker='o')
axs[1, 0].plot(df['Step'], df['Post_Sup_Acc'], color='blue', linestyle='--', label='Post_Sup_Acc', marker='o')
axs[1, 0].set_title('Sup_Acc')
axs[1, 0].set_xlabel('Step')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].legend()
axs[1, 0].grid(True)

# que_acc
axs[1, 1].plot(df['Step'], df['Pre_Que_Acc'], color='red', linestyle='-', label='Pre_Que_Acc', marker='o')
axs[1, 1].plot(df['Step'], df['Post_Que_Acc'], color='blue', linestyle='-', label='Post_Que_Acc', marker='o')
axs[1, 1].set_title('Que_Acc')
axs[1, 1].set_xlabel('Step')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('metrics_plot_tiered.png')
