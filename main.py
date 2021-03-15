import numpy as np
import tensorflow as tf


def spot_count(ndarray):
    """
    Count the number of spots in a 8bit, maximum intensity projection of fluorescently labeled spindle pole bodies
    :param ndarray:
    :return: A list of the number of spindle pole body foci in each image
    """

    from skimage.feature import blob_log

    spot_count_list = []
    for i in range(ndarray.shape[0]):
        plane = ndarray[i, :, :]
        blob_array = blob_log(plane, min_sigma=1, max_sigma=3)
        spot_count_list.append(blob_array.shape[0])

    return spot_count_list


npz = np.load('josh_input_target_output.npz')
input = npz['input']
output = npz['output']
target = npz['target']
# convert output and target ndarrays
output_tensor = tf.convert_to_tensor(output)
output_tensor = tf.expand_dims(output_tensor, axis=-1)
output_tensor = tf.cast(output_tensor, tf.uint8)
target_tensor = tf.convert_to_tensor(target)
target_tensor = tf. expand_dims(target_tensor, axis=-1)
target_tensor = tf.cast(target_tensor, tf.uint8)
ssim_tensor = tf.image.ssim(output_tensor, target_tensor, max_val=255)
ssim_array = ssim_tensor.numpy()
count_list = spot_count(input)

with open('cse4_spbcount_ssim.csv', 'w') as f:
    for count, ssim in zip(count_list, ssim_array):
        string = str(count) + ',' + str(ssim) + '\n'
        f.write(string)

f.close()
