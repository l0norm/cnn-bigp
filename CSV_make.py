import os
import csv

def generate_csv(directory, output_file="reciters.csv"):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "fname", "File Path"])
        
        for reciter in os.listdir(directory):
            reciter_path = os.path.join(directory, reciter)
            if os.path.isdir(reciter_path):  # Check if it's a folder
                i = 0
                for filename in os.listdir(reciter_path):
                    if filename.endswith(".wav"):  # Only process .wav file
                        writer.writerow([i, filename,reciter])
                        i+=1
    

    print(f"CSV file '{output_file}' generated successfully.")

# Usage example
directory_path = "C:\\Users\\fahad\\Desktop\\tests\\dataset"  # Change this to your folder path
generate_csv(directory_path)
