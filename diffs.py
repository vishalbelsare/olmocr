import json
import random

input_file = "output_multi_columns/multi_columns.jsonl"   # Replace with your input file name
output_file = "output.jsonl" # Replace with your desired output file name

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        
        # Skip lines where checked is rejected
        if data.get("checked") == "rejected":
            continue
        
        # Update max_diffs if greater than 4
        if data.get("max_diffs", 0) > 4:
            data["max_diffs"] = random.randint(1, 4)
        
        # Write the updated record to the output file as a JSON string
        outfile.write(json.dumps(data) + "\n")