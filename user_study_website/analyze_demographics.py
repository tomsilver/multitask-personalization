import json
from collections import Counter
import statistics

def analyze_demographics(file_path):
    # Read and parse the JSON data
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    # Extract participant info
    participants = [entry['participantInfo'] for entry in data]
    
    # Calculate age statistics
    ages = [int(p['age']) for p in participants]
    age_stats = {
        'count': len(ages),
        'mean': statistics.mean(ages),
        'median': statistics.median(ages),
        'min': min(ages),
        'max': max(ages)
    }
    
    # Count gender distribution
    gender_counts = Counter(p['gender'] for p in participants)
    
    # Count robot experience
    robot_exp_counts = Counter(p['robotExp'] for p in participants)
    
    # Count feeding experience
    fed_exp_counts = Counter(p['fedExp'] for p in participants)
    
    # Print summary
    print("\nDemographic Summary:")
    print("===================")
    print(f"\nTotal Participants: {age_stats['count']}")
    
    print("\nAge Statistics:")
    print(f"Mean Age: {age_stats['mean']:.1f} years")
    print(f"Median Age: {age_stats['median']} years")
    print(f"Age Range: {age_stats['min']} - {age_stats['max']} years")
    
    print("\nGender Distribution:")
    for gender, count in gender_counts.items():
        print(f"{gender}: {count} ({count/len(participants)*100:.1f}%)")
    
    print("\nRobot Experience:")
    for exp, count in robot_exp_counts.items():
        print(f"{exp}: {count} ({count/len(participants)*100:.1f}%)")
    
    print("\nFeeding Assistance Experience:")
    for exp, count in fed_exp_counts.items():
        print(f"{exp}: {count} ({count/len(participants)*100:.1f}%)")

if __name__ == "__main__":
    analyze_demographics("user_data_example.txt") 