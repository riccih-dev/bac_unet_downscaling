import tensorflow as tf

class UNetModelConfiguration:
    def configure_optimizer(self, learning_rate_value):
        """
        Configure and return the optimizer for the U-Net model.

        Parameters:
        ----------
        - learning_rate_value: float, learning rate for the optimizer.

        Returns:
        --------
        - tf.keras.optimizers.Optimizer
            Configured Adam optimizer with a specific learning rate.
        """
        return tf.optimizers.legacy.Adam(learning_rate=learning_rate_value)

    def configure_callbacks(self, scheduler_type = 'exponential_decay'):
        """
        Configure and return the callbacks for the U-Net model training.

        Parameters:
        ----------
        - scheduler_type: str, optional
            Type of learning rate scheduler. Default is 'exponential_decay'.

        Returns:
        --------
        - List of tf.keras.callbacks.Callback
            List containing LearningRateScheduler and EarlyStopping callbacks.
        """
        lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: self._lr_scheduler(epoch, lr, scheduler_type))
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        return [lr_scheduler_callback, early_stopping]
    
    
    def _lr_scheduler(self, epoch, lr, scheduler_type):
        """
        Learning rate scheduler based on the specified type.

        Parameters:
        -----------
        - epoch: int, current epoch.
        - lr: float, current learning rate.
        - scheduler_type: str, type of learning rate scheduler.

        Returns:
        --------
        - float
            Updated learning rate.
        """
        if scheduler_type == 'constant':
            return lr
        elif scheduler_type == 'step_decay':
            return self._lr_scheduler_step_decay(epoch, lr)
        elif scheduler_type == 'exponential_decay':
            return self._lr_scheduler_exponential_decay(epoch, lr)
        elif scheduler_type == 'time_decay':
            return self._lr_scheduler_time_decay(epoch, lr)
        else:
            raise ValueError(f"Invalid scheduler_type: {scheduler_type}")


    def _lr_scheduler_step_decay(self, epoch, lr):
        """
        Learning rate scheduler that reduces the learning rate by a factor after a certain number of epochs.

        Parameters:
        -----------
        - epoch: int, current epoch.
        - lr: float, current learning rate.

        Returns:
        --------
        - float
            Updated learning rate.
        """
        if epoch < 10:
            return lr
        else:
            return lr * 0.1


    def _lr_scheduler_exponential_decay(self, epoch, lr):
        """
        Learning rate scheduler that exponentially decays the learning rate after a certain number of epochs.

        Parameters:
        -----------
        - epoch: int, current epoch.
        - lr: float, current learning rate.

        Returns:
        --------
        - float
            Updated learning rate.
        """
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


    def _lr_scheduler_time_decay(self, epoch, lr):
        """
        Learning rate scheduler that decays the learning rate over time.

        Parameters:
        -----------
        - epoch: int, current epoch.
        - lr: float, current learning rate.

        Returns:
        --------
        - float
            Updated learning rate.
        """
        return lr * 1.0 / (1.0 + 0.1 * epoch)
    
    def generate_filters(self, initial_filter, num_values=4):
        """
        Generate a list of filter values based on a specified pattern.

        Parameters:
        - initial_filter (int): The initial number of filters.
        - num_values (int): The number of values to generate. Default is 5.

        Returns:
        - List[int]: A list of filter values.
        """
        generate_filter_values = lambda x: x * 2 if x <= initial_filter * (2 ** (num_values - 1)) else x / 2
        filters = [initial_filter]+[int(generate_filter_values(initial_filter * (2 ** i))) for i in range(num_values)]

        return filters


