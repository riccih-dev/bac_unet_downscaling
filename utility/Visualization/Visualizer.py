import matplotlib.pyplot as plt

class Visualizer:
    """Visualize evaluation results."""
    
    def plot_evaluation(self, metric_values, metric_name):
        """Plot evaluation metric over epochs."""
        plt.plot(metric_values)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Over Epochs')
        plt.show()