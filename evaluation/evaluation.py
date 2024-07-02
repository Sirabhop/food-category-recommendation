import numpy as np

def get_label_and_probability(predictions):
    final_results = []
    for prediction in predictions:
        max_index = np.argmax(prediction)
        probability = prediction[max_index]
        final_results.append((max_index, probability))
    return final_results