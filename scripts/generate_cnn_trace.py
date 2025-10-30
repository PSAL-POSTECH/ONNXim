import csv
import random

# Number of trace entries
num_rows = 20

# Initialize CSV
with open('./traces/resnet18.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['time', 'prompt_length', 'target_length', 'cached_length'])
    
    current_time = 0
    for _ in range(num_rows):
        # Simulate ResNet18 input
        batch_size = random.choice([1, 2, 4, 8])
        input_pixels = 224 * 224 * 3  # total pixels in image
        prompt_length = input_pixels * batch_size
        target_length = 1000  # number of output classes in ImageNet
        cached_length = 0  # ResNet18 usually doesn't use cached states
        
        writer.writerow([current_time, prompt_length, target_length, cached_length])
        
        # Increment time by random interval (simulate requests)
        current_time += random.randint(50, 200)  # ms
