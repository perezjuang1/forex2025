import os
import time
from datetime import datetime
import glob

def monitor_logs():
    """Monitor trading engine logs in real-time"""
    log_dir = 'logs'
    
    # Find all today's log files
    today = datetime.now().strftime('%Y%m%d')
    log_pattern = os.path.join(log_dir, f'robot_price_*_{today}.log')
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print(f"No log files found for today ({today})")
        return
    
    print(f"Monitoring {len(log_files)} log file(s):")
    for log_file in log_files:
        print(f"  - {log_file}")
    print("\n" + "="*80)
    print("Press Ctrl+C to stop monitoring\n")
    
    # Track file positions
    file_positions = {f: os.path.getsize(f) if os.path.exists(f) else 0 for f in log_files}
    
    try:
        while True:
            for log_file in log_files:
                if not os.path.exists(log_file):
                    continue
                
                current_size = os.path.getsize(log_file)
                last_position = file_positions.get(log_file, 0)
                
                if current_size > last_position:
                    # Read new content
                    with open(log_file, 'r', encoding='utf-8') as f:
                        f.seek(last_position)
                        new_content = f.read()
                        if new_content.strip():
                            print(f"\n[{os.path.basename(log_file)}]")
                            print(new_content, end='')
                    file_positions[log_file] = current_size
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_logs()



