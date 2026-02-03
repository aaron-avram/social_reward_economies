"""Quick script to analyze debug_follower_decisions.txt and extract key information"""

import re

log_file = "debug_follower_decisions.txt"

# Find examples of follower condition checks
print("Analyzing debug logs...\n")

with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
# Look for follower condition check sections
print("Sample follower condition checks (first 5):\n")
count = 0
i = 0
while i < len(lines) and count < 5:
    if "follower condition check" in lines[i]:
        # Print the next 30 lines to see the data
        print(f"\n{'='*80}")
        print(f"Found at line {i+1}")
        for j in range(i, min(i+35, len(lines))):
            print(lines[j].rstrip())
        print(f"{'='*80}\n")
        count += 1
        i += 35
    i += 1

# Count how many times conditions passed vs failed
all_pass_true = 0
all_pass_false = 0
switch_called = 0

for line in lines:
    if "switch_influencer called" in line:
        switch_called += 1
    if "all_conditions_pass: True" in line:
        all_pass_true += 1
    if "all_conditions_pass: False" in line:
        all_pass_false += 1

print(f"\nSummary:")
print(f"  switch_influencer called: {switch_called} times")
print(f"  all_conditions_pass: True: {all_pass_true} times")
print(f"  all_conditions_pass: False: {all_pass_false} times")










