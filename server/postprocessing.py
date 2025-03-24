import re
import argparse

def remove_stablehlo_prefix(input_file, output_file):
    # Read the .dot file
    with open(input_file, "r") as file:
        content = file.read()

    # Use regex to remove the 'stablehlo.' prefix
    modified_content = re.sub(r"stablehlo\.", "", content)

    # Write the modified content to the output file
    with open(output_file, "w") as file:
        file.write(modified_content)

def main():
    parser = argparse.ArgumentParser(description='Remove stablehlo. prefix from dot files.')
    parser.add_argument('--input', default="viz.dot", help='Input dot file path (default: viz.dot)')
    parser.add_argument('--output', help='Output dot file path (default: same as input)')
    
    args = parser.parse_args()
    
    # If output is not specified, use the input file
    output_file = args.output if args.output else args.input
    
    remove_stablehlo_prefix(args.input, output_file)
    
    print(f"Processed {args.input} and saved to {output_file}")

if __name__ == "__main__":
    main()
