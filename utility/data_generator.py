import numpy as np

class DataGenerator:
    def __init__(self, data, labels, batch_size, shuffle=True): 
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data['time'])
        self.indices = np.arange(self.num_samples)
        self.on_epoch_end()                                   

    def on_epoch_end(self):
      '''
      shuffles indices at the end of each epoch. Shuffling depends on var shuffle. If set to True, the indices will be randomly shuffled
      to introduce variability in the order of samples seen by the model during training.
      This is typically done to avoid the model learning patterns specific to the order of the training data.
      '''
      if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # Return the number of batches in one epoch
        return int(np.ceil(self.num_samples / self.batch_size))

    def generate_batches(self):
        self.on_epoch_end()
        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            batch_indices = self.indices[start:end]

            # Select the subset of time corresponding to the batch
            batch_data = self.data.isel(time=batch_indices)
            batch_labels = self.labels.isel(time=batch_indices)

            # Convert xarray Datasets to numpy arrays
            batch_features = np.stack([batch_data[var].values for var in batch_data.data_vars], axis=-1)
            batch_labels = np.stack([batch_labels[var].values for var in batch_labels.data_vars], axis=-1)

            yield batch_features, batch_labels
