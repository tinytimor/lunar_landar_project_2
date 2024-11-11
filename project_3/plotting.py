import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric(data, layout_avg_data, x_col, y_col, color, sma_color, sma_window=100, label=None, sma_label=None, save_path=None):
    plt.figure(figsize=(12, 6))
    
    #  getting the raw data for plotting
    sns.lineplot(data=data, x=x_col, y=y_col, color=color, alpha=0.3, label=label)
    
    # calculating and plotting the moving avage
    sma = data[y_col].rolling(window=sma_window, min_periods=1).mean()
    sns.lineplot(x=data[x_col], y=sma, color=sma_color, linewidth=2, label=sma_label)
    
    # Plot the layout-wise 3000-episode bin averages
    sns.lineplot(data=layout_avg_data, x='episode_bin', y=y_col, hue='layout', linewidth=2, linestyle='--', marker='o', legend='brief')

    plt.xlabel(x_col.capitalize())
    plt.ylabel(y_col.replace("_", " ").capitalize())
    plt.title(f"{y_col.replace('_', ' ').capitalize()} Over {x_col.capitalize()} (with 3000-Episode Moving Average)")
    plt.legend()
    plt.tight_layout()

    # Save the plot if a save path is specified
    if save_path:
        plt.savefig(save_path, format='png')
        print(f"Plot saved as {save_path}")

    plt.show()
