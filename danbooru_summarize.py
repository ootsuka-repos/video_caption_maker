import os
from collections import Counter

# Define the directory containing the captions.txt files
base_dir = r"C:\Users\user\Desktop\git\video_dataset_trim\outputs\captions"

def count_tags_in_file(file_path):
    tag_counter = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            tags = line.strip().split(", ")
            tag_counter.update(tags)
    return tag_counter, len(lines)

def find_and_filter_tags(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "captions.txt":
                file_path = os.path.join(root, file)
                tag_counts, total_lines = count_tags_in_file(file_path)
                
                # Calculate the threshold for 70% of the total lines
                threshold = total_lines * 0.7
                
                # Filter tags that appear more than 70% of the time
                selected_tags = [tag for tag, count in tag_counts.items() if count > threshold]
                
                # Write the selected tags to a new file
                new_file_path = os.path.join(root, "selected_tags.txt")
                with open(new_file_path, "w", encoding="utf-8") as f:
                    f.write(", ".join(selected_tags))
                
                print(f"File: {file_path}")
                print(f"Selected tags: {', '.join(selected_tags)}\n")

# Find and filter tags in all captions.txt files in the base directory
find_and_filter_tags(base_dir)