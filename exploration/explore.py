import json
import pandas as pd

file_path = "../data/raw/raw_subset/amplitude_export_chunk_1_anonymized_subset.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # Load entire JSON array

# Print first few records to verify
for i, record in enumerate(data[:3]):  # Display first 3 records
    print(f"Record {i+1}:", json.dumps(record, indent=4))


# chunks = []
# with open("large_data.json", "r") as f:
#     for line in f:
#         record = json.loads(line)
#         flattened = flatten_json(record)
#         chunks.append(flattened)
#         if len(chunks) == 10000:  # Process in batches of 10K
#             pd.DataFrame(chunks).to_csv("structured_data.csv", mode="a", index=False, header=False)
#             chunks = []  # Clear memory
