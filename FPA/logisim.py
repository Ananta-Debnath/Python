import os
import subprocess

LOGISIM_JAR = 'D:\Ananta\Academic\BUET\L2-T2\CSE210\Logisim\Logisim-Executables\logisim-generic-2.7.1.jar'  # Path to the Logisim JAR file
CIRCUIT_FILE = "D:\Ananta\Academic\BUET\L2-T2\CSE210\Assignment2\FPA_Test.circ"


def run_with_timeout():
    command = ['java', '-jar', LOGISIM_JAR, '-tty', 'table', CIRCUIT_FILE]

    try:
        # Let Logisim run for exactly 1 second, then forcefully kill it
        subprocess.run(command, capture_output=True, text=True, timeout=1)
        
    except subprocess.TimeoutExpired as e:
        # e.stdout contains all the text printed before the timeout!
        print("Logisim forcefully stopped after 1 second.")
        print("Here is what it managed to output:")
        
        # e.stdout is a string containing the table
        output_text = e.stdout 
        print(output_text)
        
# run_with_timeout()




def test_circuit(hex_val_a, hex_val_b):
    # 1. Read your prepped circuit file
    with open(CIRCUIT_FILE, 'r') as file:
        circuit_data = file.read()
    
    # 2. Swap out our placeholder constants with the new test values
    circuit_data = circuit_data.replace('0xaaaa', hex_val_a)
    circuit_data = circuit_data.replace('0xbbbb', hex_val_b)
    
    # 3. Save as a temporary file so we don't overwrite your original
    temp_file = 'temp_run.circ'
    with open(temp_file, 'w') as file:
        file.write(circuit_data)
        
    # 4. Run Logisim with the TIMEOUT approach
    command = ['java', '-jar', LOGISIM_JAR, '-tty', 'table', temp_file]
    
    try:
        # Run for 1 second, then forcefully kill it
        subprocess.run(command, capture_output=True, text=True, timeout=1)
        
    except subprocess.TimeoutExpired as e:
        # e.stdout contains everything printed before it was killed
        output_text = e.stdout 
        
        print(f"\n--- Testing A = {hex_val_a} | B = {hex_val_b} ---")
        
        # Clean up the output by splitting it into lines and removing blanks
        lines = [line.strip() for line in output_text.split('\n') if line.strip()]
        
        # Usually, line 0 is the pin names (header), and line 1 is the binary result
        if len(lines) >= 2:
            print("Pins:  ", lines[0])
            print("Result:", lines[1])
        else:
            print("Raw Output:\n", output_text)

# Test your circuit with some 16-bit hex numbers!
# (Make sure to always format them as '0x' followed by 4 characters)
# test_circuit('0x0005', '0x0002')
# test_circuit('0x000A', '0x000A')
# test_circuit('0xFFFF', '0x0001')



def validate_through_logisim(txt_file):
    # 1. Load the template circuit (Make sure 0xaaaa and 0xbbbb are saved in this file!)
    with open(CIRCUIT_FILE, 'r') as file:
        base_circuit = file.read()

    passed_count = 0
    failed_count = 0

    with open(txt_file, 'r') as file:
        lines = file.readlines()

    print(f"Starting validation for {len(lines)} tests...\n")

    for line_num, line in enumerate(lines, 1):
        # Split the line into individual binary strings
        parts = line.strip().split()
        if len(parts) < 3:
            continue # Skip empty or malformed lines

        # Extract inputs and expected outputs
        bin_input_a = parts[0]
        bin_input_b = parts[1]
        expected_outputs = parts[2:] # Grabs all remaining items in the list

        # Convert binary string inputs to Logisim hex format (e.g., '0000000000000101' -> '0x0005')
        hex_a = f"0x{int(bin_input_a, 2):04X}"
        hex_b = f"0x{int(bin_input_b, 2):04X}"

        # 2. Swap constants
        temp_circuit = base_circuit.replace('0xaaaa', hex_a)
        temp_circuit = temp_circuit.replace('0xbbbb', hex_b)

        with open('temp_run.circ', 'w') as f:
            f.write(temp_circuit)

        # 3. Run Logisim
        command = ['java', '-jar', LOGISIM_JAR, 'temp_run.circ', '-tty', 'table']
        try:
            # Using the timeout trick since we know the circuit gets stuck
            subprocess.run(command, capture_output=True, text=True, timeout=0.6)
            
        except subprocess.TimeoutExpired as e:
            output_text = e.stdout
            
            # 4. Clean up the output
            # Grab the very first line of text Logisim managed to print
            first_line = output_text.strip().split('\n')[0]
            
            if first_line:
                # Remove the spaces Logisim puts between every 4 bits 
                # (e.g., "0000 1111" becomes "00001111")
                no_spaces_line = first_line.replace(" ", "")
                
                # Now split by tabs to separate the individual output pins
                logisim_actual_outputs = no_spaces_line.split('\t')
                
                # Clean up any empty strings left over just in case
                logisim_actual_outputs = [val for val in logisim_actual_outputs if val]
                
                # 5. Compare the results!
                if logisim_actual_outputs == expected_outputs:
                    print(f"Line {line_num} [PASS]: {hex_a} + {hex_b}")
                    passed_count += 1
                else:
                    print(f"Line {line_num} [FAIL]: {hex_a} + {hex_b}")
                    print(f"  Expected: {expected_outputs}")
                    print(f"  Got:      {logisim_actual_outputs}")
                    failed_count += 1
            else:
                print(f"Line {line_num} [ERROR]: Logisim didn't output any data.")
    print("\n" + "="*30)
    print(f"VALIDATION COMPLETE")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("="*30)

# Run the validator (Change 'tests.txt' to the name of your file)
# find files of .txt in FPA/
files = [os.path.join('FPA', f) for f in os.listdir('FPA') if f.endswith('.txt')]

for filename in files:
    print(f"\nValidating test file: {filename}")
    validate_through_logisim(filename)