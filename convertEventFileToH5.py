from tensorboard.backend.event_processing import event_accumulator
import h5py

def convertEventFileToH5(event_file_path, hdf5_file_path):

    # Create an event accumulator to read the event file
    event_acc = event_accumulator.EventAccumulator(event_file_path)
    event_acc.Reload()

    # Create an HDF5 file to store the metrics
    with h5py.File(hdf5_file_path, 'w') as h5_file:
        for tag in event_acc.Tags()["scalars"]:
            # Get the data for the scalar metric
            data = event_acc.Scalars(tag)

            # Extract step and value for each data point
            steps = [item.step for item in data]
            values = [item.value for item in data]

            # Create datasets in the HDF5 file
            h5_file.create_dataset(f'steps/{tag}', data=steps)
            h5_file.create_dataset(f'values/{tag}', data=values)


if __name__ == "__main__":
    # Path to your TensorBoard event file
    event_file_path = 'checkpoints/training_2023.09.29.15.55.42/lightning_logs/version_0/events.out.tfevents.1696002943.abadea-notebook-1.15128.0'
    # Define the path to the HDF5 file where you want to save the metrics
    hdf5_file_path = 'metrics.h5'
    convertEventFileToH5(event_file_path, hdf5_file_path)
