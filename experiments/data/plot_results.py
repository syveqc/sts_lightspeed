import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_losses_from_npy(directory, filter=None):
    """Loads .npy files containing loss values into a pandas DataFrame."""
    data = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.npy'):
            if filter is not None:
                contained = False
                for name in filter:
                    if name in file_name:
                        contained = True
                        break
                if not contained:
                    continue
            # Split from the right to ensure we only separate the last underscore
            experiment_name, timestamp = file_name[:-4].rsplit('_', 1)
            losses = np.load(os.path.join(directory, file_name))
            # Create a DataFrame for each file, including a timestep (index) column
            df = pd.DataFrame({
                'loss': losses,
                'experiment': experiment_name,
                'timestep': np.arange(len(losses))  # Create a timestep column based on array index
            })
            data.append(df)
    return pd.concat(data, ignore_index=True)

def smooth_losses(df, window_size=10):
    """Applies a rolling average (smoothing) to the loss values for each experiment."""
    smoothed_df = df.copy()
    smoothed_df['smoothed_loss'] = smoothed_df.groupby('experiment')['loss'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    return smoothed_df

def plot_losses_with_confidence_intervals(df):
    """Plots smoothed losses over timesteps with confidence intervals using seaborn."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='timestep', y='smoothed_loss', hue='experiment', data=df, errorbar='sd')
    plt.title('Smoothed Losses over Timesteps with Confidence Intervals')
    plt.xlabel('Timestep')
    plt.ylabel('Smoothed Loss')
    plt.ylim(0.035, 0.07)
    plt.tight_layout()
    plt.savefig('losses_2048.jpg')

# Example usage
directory = 'results'  # Update this with your actual directory path
# df = load_losses_from_npy(directory, ['one_hot_1024', 'more_dropout'])
df = load_losses_from_npy(directory, ['2048'])
df_smoothed = smooth_losses(df, window_size=100)  # Apply smoothing with window size of 10
plot_losses_with_confidence_intervals(df_smoothed)

