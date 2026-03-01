import os
import matplotlib
matplotlib.use('AGG')  # Non-GUI backend
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'training_plots')


def generate_training_graphs(combined_history, epoch_times):
    """
    Generate Accuracy and Time-per-Epoch graphs from training history.
    
    Args:
        combined_history: dict with keys 'accuracy', 'val_accuracy' (lists per epoch)
        epoch_times: list of time in seconds per epoch
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    _plot_accuracy(combined_history)
    _plot_time(epoch_times)

    print(f"[SAVED] Training graphs saved to: {PLOTS_DIR}")


def _plot_accuracy(history):
    """Training vs Validation Accuracy across all epochs."""
    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['accuracy'], 'b-o', label='Training Accuracy', markersize=4)
    plt.plot(epochs, history['val_accuracy'], 'r-o', label='Validation Accuracy', markersize=4)

    # Draw a vertical line between Phase 1 and Phase 2
    phase1_epochs = len(history['accuracy']) // 2
    plt.axvline(x=phase1_epochs + 0.5, color='gray', linestyle='--', alpha=0.7, label='Phase 1 → Phase 2')

    plt.title('Model Accuracy Over Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy.png'), dpi=150)
    plt.close()
    print("[GRAPH] accuracy.png generated")


def _plot_time(epoch_times):
    """Time taken per epoch."""
    epochs = range(1, len(epoch_times) + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(epochs, epoch_times, color='#3498db', alpha=0.8, edgecolor='#2980b9')
    plt.plot(epochs, epoch_times, 'r-o', markersize=4, label='Time trend')

    # Draw a vertical line between Phase 1 and Phase 2
    phase1_epochs = len(epoch_times) // 2
    plt.axvline(x=phase1_epochs + 0.5, color='gray', linestyle='--', alpha=0.7, label='Phase 1 → Phase 2')

    avg_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    plt.axhline(y=avg_time, color='green', linestyle=':', alpha=0.7, label=f'Average ({avg_time:.1f}s)')

    plt.title('Time Per Epoch', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'time_per_epoch.png'), dpi=150)
    plt.close()
    print("[GRAPH] time_per_epoch.png generated")
