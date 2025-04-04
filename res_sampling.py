import random

def reservoir_sample(input_path, output_path, sample_size=3000):
    reservoir = []
    with open(input_path, 'r') as infile:
        random.seed(17)
        for i, line in enumerate(infile):
            if i < sample_size:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line

    with open(output_path, 'w') as outfile:
        outfile.writelines(reservoir)

# Example usage
reservoir_sample('s2pdf_paths.txt', 'sampled_image_links_order.txt', sample_size=3000)